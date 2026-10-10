// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include <chrono>
#include <cstdio>
#include <mutex>
#include <utility>
#include <vector>

#include "common/Console.h"
#include "common/Timer.h"
#include "common/Threading.h"

#include "PerformanceMetrics.h"

#include "F7Fork.h"
#include "GS.h"
#include "MTGS.h"
#include "MTVU.h"
#include "VMManager.h"

#include <algorithm>
#include <atomic>

#if defined(__ANDROID__)
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static const float UPDATE_INTERVAL = 0.5f;

static float s_fps = 0.0f;
static PerformanceMetrics::AverageFPS s_avg_vps;
static float s_internal_fps = 0.0f;
static float s_minimum_frame_time = 0.0f;
static float s_minimum_frame_time_accumulator = 0.0f;
static float s_average_frame_time = 0.0f;
static float s_average_frame_time_accumulator = 0.0f;
static float s_maximum_frame_time = 0.0f;
static float s_maximum_frame_time_accumulator = 0.0f;
static u32 s_frames_since_last_update = 0;
static u32 s_unskipped_frames_since_last_update = 0;
static Common::Timer s_last_update_time;
static Common::Timer s_last_frame_time;

// Session perf logging: a rolling emulog line every ~30s of presented
// frames, and a whole-session average at shutdown (LogSessionSummary).
// Gives every -logfile run a durable framerate record. Wall-clock based:
// paused time dilutes the session average but not the rolling lines.
static const float LOG_INTERVAL = 30.0f;
static Common::Timer s_session_timer;
static float s_log_accum_time = 0.0f;
static u32 s_log_accum_frames = 0;

// frame number, updated by the GS thread
static u64 s_frame_number = 0;

// internal fps heuristics
static PerformanceMetrics::InternalFPSMethod s_internal_fps_method = PerformanceMetrics::InternalFPSMethod::None;
static u32 s_gs_framebuffer_blits_since_last_update = 0;
static u32 s_gs_privileged_register_writes_since_last_update = 0;

static Threading::ThreadHandle s_cpu_thread_handle;
static u64 s_last_cpu_time = 0;
static u64 s_last_gs_time = 0;
static u64 s_last_gs_back_time = 0;
static u64 s_last_vu_time = 0;
static u64 s_last_ticks = 0;

// The GS back thread registers itself from its own entry point and is cleared from the MTGS
// thread after the join, so unlike every other handle here this one is written by a thread
// other than the one sampling it. Handle and running total are therefore both owned by the
// mutex, which is taken twice a second at most.
static std::mutex s_gs_back_thread_mutex;
static Threading::ThreadHandle s_gs_back_thread_handle;

/// CPU time consumed by the back thread since the last call, in Threading tick units.
/// A thread that started or stopped inside the window contributes only the part of it that
/// the handle was installed for; the alternative is a u64 subtraction that wraps into a
/// nonsense percentage on the first window after a GSreopen.
static u64 SampleGSBackThreadCPUTime()
{
	std::unique_lock lock(s_gs_back_thread_mutex);
	const u64 now = s_gs_back_thread_handle ? s_gs_back_thread_handle.GetCPUTime() : s_last_gs_back_time;
	const u64 delta = (now > s_last_gs_back_time) ? (now - s_last_gs_back_time) : 0;
	s_last_gs_back_time = now;
	return delta;
}

#if defined(__ANDROID__)
// ---- Android ADPF (PerformanceHintManager) ---------------------------------
// Tells the OS "these threads produce a frame every N ns, clock them to hit it."
// PS2 emulation is bursty and latency-sensitive, so Android's DVFS governor
// routinely under-clocks the CPU/GPU under it; ADPF is the purpose-built fix.
// Symbols are resolved at runtime from libandroid.so so the app still loads on
// pre-API-33 devices (there the session is never created — everything no-ops).
struct AdpfApi
{
	void* (*getManager)() = nullptr;
	void* (*createSession)(void*, const int32_t*, size_t, int64_t) = nullptr;
	int (*updateTarget)(void*, int64_t) = nullptr;
	int (*reportActual)(void*, int64_t) = nullptr;
	void (*closeSession)(void*) = nullptr;
	bool tried = false;

	bool Available()
	{
		if (!tried)
		{
			tried = true;
			if (void* lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL))
			{
				getManager = reinterpret_cast<decltype(getManager)>(dlsym(lib, "APerformanceHint_getManager"));
				createSession = reinterpret_cast<decltype(createSession)>(dlsym(lib, "APerformanceHint_createSession"));
				updateTarget = reinterpret_cast<decltype(updateTarget)>(dlsym(lib, "APerformanceHint_updateTargetWorkDuration"));
				reportActual = reinterpret_cast<decltype(reportActual)>(dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
				closeSession = reinterpret_cast<decltype(closeSession)>(dlsym(lib, "APerformanceHint_closeSession"));
			}
		}
		return getManager && createSession && updateTarget && reportActual && closeSession;
	}
};

static std::mutex s_adpf_mutex;
static AdpfApi s_adpf;
static std::vector<int32_t> s_adpf_tids;
static void* s_adpf_manager = nullptr;
static void* s_adpf_session = nullptr;
static bool s_adpf_enabled = false; // experimental, opt-in
static bool s_adpf_create_failed = false;
static bool s_adpf_report_warned = false;
static bool s_adpf_paused = false; // reporting suspended (unlimited/vsync/interrupted), edge-logged
static int64_t s_adpf_target_ns = 0;
static Common::Timer::Value s_adpf_work_start = 0; // start of the current active-work segment (0 = none)
static int64_t s_adpf_work_accum_ns = 0; // this frame's segments closed so far; -1 = frame incomplete, skip it

// The frame deadline in ns: emulated refresh scaled by the limiter's target speed, so turbo /
// slow-motion move the deadline correctly. Returns 0 when there is no finite deadline (Unlimited,
// GetTargetSpeed()==0), so the caller pauses ADPF rather than feeding a bogus/inf target.
static int64_t AdpfTargetNs()
{
	const double fps = VMManager::GetFrameRate() * static_cast<double>(VMManager::GetTargetSpeed());
	return (fps > 1.0) ? static_cast<int64_t>(1.0e9 / fps) : 0;
}

// Must hold s_adpf_mutex. Creates the session once a perf thread has registered and a finite
// deadline exists, and LOGS the real outcome — so "ACTIVE" means a session genuinely exists,
// not merely that the user flipped the toggle.
static void AdpfEnsureSession()
{
	if (s_adpf_session || s_adpf_create_failed || !s_adpf_enabled || s_adpf_tids.empty())
		return;
	if (!s_adpf.Available())
	{
		s_adpf_create_failed = true; // pre-API-33 / no libandroid; stop retrying
		Console.WriteLn("ADPF: PerformanceHintManager symbols unavailable (needs Android 13+) — clock hint inactive.");
		return;
	}
	if (!s_adpf_manager)
		s_adpf_manager = s_adpf.getManager();
	if (!s_adpf_manager)
	{
		s_adpf_create_failed = true;
		Console.WriteLn("ADPF: getManager() returned null — clock hint inactive.");
		return;
	}
	const int64_t target = AdpfTargetNs();
	if (target <= 0)
		return; // no finite deadline yet (Unlimited / VM not paced); retry once one exists
	s_adpf_session = s_adpf.createSession(s_adpf_manager, s_adpf_tids.data(), s_adpf_tids.size(), target);
	if (!s_adpf_session)
	{
		s_adpf_create_failed = true;
		Console.Warning("ADPF: createSession() over %zu threads failed — clock hint inactive.", s_adpf_tids.size());
		return;
	}
	s_adpf_target_ns = target;
	s_adpf_report_warned = false;
	s_adpf_paused = false;
	std::string tid_list;
	for (size_t i = 0; i < s_adpf_tids.size(); i++)
		tid_list += (i ? "," : "") + std::to_string(s_adpf_tids[i]);
	Console.WriteLn("ADPF: session ACTIVE over %zu threads [tids %s], deadline %.2f ms.", s_adpf_tids.size(),
		tid_list.c_str(), static_cast<double>(target) / 1.0e6);
}
#endif // __ANDROID__

static double s_cpu_thread_usage = 0.0f;
static double s_cpu_thread_time = 0.0f;
static float s_gs_thread_usage = 0.0f;
static float s_gs_thread_time = 0.0f;
static float s_gs_back_thread_usage = 0.0f;
static float s_gs_back_thread_time = 0.0f;
static float s_vu_thread_usage = 0.0f;
static float s_vu_thread_time = 0.0f;

static PerformanceMetrics::FrameTimeHistory s_frame_time_history;
static u32 s_frame_time_history_pos = 0;

struct GSSWThreadStats
{
	Threading::ThreadHandle handle;
	u64 last_cpu_time = 0;
	double usage = 0.0;
	double time = 0.0;
};
std::vector<GSSWThreadStats> s_gs_sw_threads;

static float s_average_gpu_time = 0.0f;
static float s_last_gpu_time = 0.0f;
static float s_accumulated_gpu_time = 0.0f;
static float s_gpu_usage = 0.0f;
static u32 s_presents_since_last_update = 0;
static double s_average_gpu_vs_invocations = 0.0;
static double s_average_gpu_ps_invocations = 0.0;
static u64 s_accumulated_gpu_vs_invocations = 0;
static u64 s_accumulated_gpu_ps_invocations = 0;

// ---- Poco F7 fork (P1): the F7PROF line ----------------------------------------------------------
// Every ~5 s of emulation, at the 0.5 s metric update on the MTGS thread, one console line (so it
// reaches logcat) splits the wall time of the GS front (MTGS) and GS Back threads into waits and
// the rest. The counters live in F7Fork.h; every window state below is touched only on the MTGS
// thread, except the rebase flag, which Reset() raises from the CPU thread.
static constexpr double F7PROF_INTERVAL = 5.0;

struct F7ProfPair
{
	u64 ticks = 0;
	u64 count = 0;
};

struct F7ProfSample
{
	F7ProfPair vu;
	u64 vu_packets = 0;
	F7ProfPair drain[static_cast<size_t>(F7Prof::Drain::Count)];
	F7ProfPair full[static_cast<size_t>(F7Prof::Full::Count)];
	F7ProfPair idle;
	F7ProfPair vsync_body;
	F7ProfPair gpu;
	F7ProfPair back_idle;
	F7ProfPair back_gpu;
	u64 back_records = 0;
};

static std::atomic<bool> s_f7_rebase{true};
static bool s_f7_legend_logged = false;
static F7ProfSample s_f7_base;
static double s_f7_time = 0.0;
static u32 s_f7_frames = 0;
static u64 s_f7_cpu_ee = 0;
static u64 s_f7_cpu_gs = 0;
static u64 s_f7_cpu_gsb = 0;
static u64 s_f7_cpu_vu = 0;

static F7ProfPair F7ProfRead(const F7Prof::Counter& counter)
{
	F7ProfPair pair;
	pair.ticks = counter.ticks.load(std::memory_order_relaxed);
	pair.count = counter.count.load(std::memory_order_relaxed);
	return pair;
}

static F7ProfSample F7ProfTake()
{
	F7ProfSample s;
	s.vu = F7ProfRead(F7Prof::g_front.vu);
	s.vu_packets = F7Prof::g_front.vu_packets.load(std::memory_order_relaxed);
	for (size_t i = 0; i < static_cast<size_t>(F7Prof::Drain::Count); i++)
		s.drain[i] = F7ProfRead(F7Prof::g_front.drain[i]);
	for (size_t i = 0; i < static_cast<size_t>(F7Prof::Full::Count); i++)
		s.full[i] = F7ProfRead(F7Prof::g_front.full[i]);
	s.idle = F7ProfRead(F7Prof::g_front.idle);
	s.vsync_body = F7ProfRead(F7Prof::g_front.vsync_body);
	s.gpu = F7ProfRead(F7Prof::g_front.gpu);
	s.back_idle = F7ProfRead(F7Prof::g_back.idle);
	s.back_gpu = F7ProfRead(F7Prof::g_back.gpu);
	s.back_records = F7Prof::g_back.records.load(std::memory_order_relaxed);
	return s;
}

static void F7ProfResetWindow(const F7ProfSample& base)
{
	s_f7_base = base;
	s_f7_time = 0.0;
	s_f7_frames = 0;
	s_f7_cpu_ee = 0;
	s_f7_cpu_gs = 0;
	s_f7_cpu_gsb = 0;
	s_f7_cpu_vu = 0;
}

static void F7ProfLogWindow(const F7ProfSample& now)
{
	const F7ProfSample& base = s_f7_base;
	const double frames = static_cast<double>(s_f7_frames);
	const double tick_ms = 1000.0 / static_cast<double>(GetTickFrequency());
	const double thread_ms = 1000.0 / static_cast<double>(Threading::GetThreadTicksPerSecond());

	// Everything per emulated field (vsync): ms per field, and events per field.
	const auto ms = [&](const F7ProfPair& n, const F7ProfPair& b) {
		return static_cast<double>(n.ticks - b.ticks) * tick_ms / frames;
	};
	const auto per = [&](u64 n, u64 b) { return static_cast<double>(n - b) / frames; };
	const auto cpu = [&](u64 thread_ticks) { return static_cast<double>(thread_ticks) * thread_ms / frames; };

	const double wall = s_f7_time * 1000.0 / frames;

	double drain_ms[static_cast<size_t>(F7Prof::Drain::Count)];
	double drain_n[static_cast<size_t>(F7Prof::Drain::Count)];
	double drain_total_ms = 0.0, drain_total_n = 0.0;
	for (size_t i = 0; i < static_cast<size_t>(F7Prof::Drain::Count); i++)
	{
		drain_ms[i] = ms(now.drain[i], base.drain[i]);
		drain_n[i] = per(now.drain[i].count, base.drain[i].count);
		drain_total_ms += drain_ms[i];
		drain_total_n += drain_n[i];
	}

	double full_ms[static_cast<size_t>(F7Prof::Full::Count)];
	double full_total_ms = 0.0, full_total_n = 0.0;
	for (size_t i = 0; i < static_cast<size_t>(F7Prof::Full::Count); i++)
	{
		full_ms[i] = ms(now.full[i], base.full[i]);
		full_total_ms += full_ms[i];
		full_total_n += per(now.full[i].count, base.full[i].count);
	}

	const double vu_ms = ms(now.vu, base.vu);
	const double gpu_ms = ms(now.gpu, base.gpu);
	const double idle_ms = ms(now.idle, base.idle);
	const double front_rest = std::max(0.0, wall - (vu_ms + drain_total_ms + full_total_ms + gpu_ms + idle_ms));

	constexpr size_t OTHER = static_cast<size_t>(F7Prof::Drain::Other);
	constexpr size_t VSYNC = static_cast<size_t>(F7Prof::Drain::Vsync);
	constexpr size_t FIFO = static_cast<size_t>(F7Prof::Drain::Fifo);
	constexpr size_t AA1 = static_cast<size_t>(F7Prof::Drain::AA1);
	constexpr size_t MOVE = static_cast<size_t>(F7Prof::Drain::Move);

	char line[768];
	int len = std::snprintf(line, sizeof(line),
		"F7PROF %.1fs %uf %.1fvps | GS vu %.2f/%.1f(%.1fpk) drn %.2f/%.1f(vs %.2f/%.1f rb %.2f/%.1f "
		"aa %.2f/%.1f mv %.2f/%.1f ot %.2f/%.1f) full %.2f/%.1f(r %.2f d %.2f p %.2f) gpu %.2f/%.1f "
		"idle %.2f rest %.2f vsb %.2f cpu %.2f",
		s_f7_time, s_f7_frames, frames / s_f7_time,
		vu_ms, per(now.vu.count, base.vu.count), per(now.vu_packets, base.vu_packets),
		drain_total_ms, drain_total_n,
		drain_ms[VSYNC], drain_n[VSYNC], drain_ms[FIFO], drain_n[FIFO],
		drain_ms[AA1], drain_n[AA1], drain_ms[MOVE], drain_n[MOVE], drain_ms[OTHER], drain_n[OTHER],
		full_total_ms, full_total_n,
		full_ms[static_cast<size_t>(F7Prof::Full::Ring)], full_ms[static_cast<size_t>(F7Prof::Full::DrawPool)],
		full_ms[static_cast<size_t>(F7Prof::Full::PayloadPool)],
		gpu_ms, per(now.gpu.count, base.gpu.count),
		idle_ms, front_rest, ms(now.vsync_body, base.vsync_body), cpu(s_f7_cpu_gs));
	len = std::clamp(len, 0, static_cast<int>(sizeof(line)) - 1);

	if (PerformanceMetrics::HasGSBackThread())
	{
		const double back_idle = ms(now.back_idle, base.back_idle);
		const double back_gpu = ms(now.back_gpu, base.back_gpu);
		len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len),
			" | GSB idle %.2f gpu %.2f/%.1f rest %.2f cpu %.2f rec %.0f",
			back_idle, back_gpu, per(now.back_gpu.count, base.back_gpu.count),
			std::max(0.0, wall - back_idle - back_gpu), cpu(s_f7_cpu_gsb),
			per(now.back_records, base.back_records));
	}
	else
	{
		len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len), " | GSB off");
	}
	len = std::clamp(len, 0, static_cast<int>(sizeof(line)) - 1);

	std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len), " | EE %.2f VU %.2f",
		cpu(s_f7_cpu_ee), cpu(s_f7_cpu_vu));

	if (!s_f7_legend_logged)
	{
		s_f7_legend_logged = true;
		Console.WriteLn("F7PROF legend: per emulated field (vsync); a/b = ms/count per field. GS = MTGS front: "
						"vu = wait on MTVU (pk = VU1 packets), drn = back-queue drains (vs vsync, rb FIFO "
						"readback, aa AA1, mv move hook, ot other), full = ring/pool backpressure (r ring, "
						"d draw pool, p payload pool).");
		Console.WriteLn("F7PROF legend: gpu = vkWaitForFences, idle = waiting for EE, rest = wall - (vu+drn+full+"
						"gpu+idle), vsb = vsync body (Merge/present/OSD), cpu = thread CPU. GSB = GS Back: idle = "
						"waiting for records, rest = wall - idle - gpu, rec = records. EE/VU = thread CPU ms.");
	}
	Console.WriteLn("%s", line);
}

/// Called at every 0.5 s metric update, on the MTGS thread, with that update's window.
static void F7ProfOnUpdate(double time, u32 frames, u64 cpu_ee, u64 cpu_gs, u64 cpu_gsb, u64 cpu_vu)
{
	// (Re)start the window: at boot and after Reset() (resume from pause, fast-boot end) the
	// counters kept running while the update window did not, so that stretch is dropped.
	if (s_f7_rebase.exchange(false, std::memory_order_acq_rel))
	{
		F7ProfResetWindow(F7ProfTake());
		return;
	}

	s_f7_time += time;
	s_f7_frames += frames;
	s_f7_cpu_ee += cpu_ee;
	s_f7_cpu_gs += cpu_gs;
	s_f7_cpu_gsb += cpu_gsb;
	s_f7_cpu_vu += cpu_vu;
	if (s_f7_time < F7PROF_INTERVAL || s_f7_frames == 0)
		return;

	const F7ProfSample now = F7ProfTake();
	F7ProfLogWindow(now);
	F7ProfResetWindow(now);
}

void PerformanceMetrics::Clear()
{
	Reset();

	s_fps = 0.0f;
	s_avg_vps.ClearStats();
	s_internal_fps = 0.0f;
	s_minimum_frame_time = 0.0f;
	s_average_frame_time = 0.0f;
	s_maximum_frame_time = 0.0f;
	s_internal_fps_method = PerformanceMetrics::InternalFPSMethod::None;

	s_cpu_thread_usage = 0.0f;
	s_cpu_thread_time = 0.0f;
	s_gs_thread_usage = 0.0f;
	s_gs_thread_time = 0.0f;
	s_gs_back_thread_usage = 0.0f;
	s_gs_back_thread_time = 0.0f;
	s_vu_thread_usage = 0.0f;
	s_vu_thread_time = 0.0f;

	s_average_gpu_time = 0.0f;
	s_last_gpu_time = 0.0f;
	s_gpu_usage = 0.0f;

	s_frame_number = 0;

	s_session_timer.Reset();
	s_log_accum_time = 0.0f;
	s_log_accum_frames = 0;

	s_frame_time_history.fill(0.0f);
	s_frame_time_history_pos = 0;
}

void PerformanceMetrics::LogSessionSummary()
{
	const double elapsed = s_session_timer.GetTimeSeconds();
	if (s_frame_number == 0 || elapsed < 1.0)
		return;
	Console.WriteLn("PerfLog session: %llu frames in %.1fs wall = %.2f fps average",
		static_cast<unsigned long long>(s_frame_number), elapsed,
		static_cast<double>(s_frame_number) / elapsed);
}

void PerformanceMetrics::Reset()
{
	s_frames_since_last_update = 0;
	s_unskipped_frames_since_last_update = 0;
	s_gs_framebuffer_blits_since_last_update = 0;
	s_gs_privileged_register_writes_since_last_update = 0;
	s_minimum_frame_time_accumulator = 0.0f;
	s_average_frame_time_accumulator = 0.0f;
	s_maximum_frame_time_accumulator = 0.0f;

	s_accumulated_gpu_time = 0.0f;
	s_presents_since_last_update = 0;

	s_last_update_time.Reset();
	s_last_frame_time.Reset();

	s_last_cpu_time = s_cpu_thread_handle.GetCPUTime();
	s_last_gs_time = MTGS::GetThreadHandle().GetCPUTime();
	SampleGSBackThreadCPUTime(); // rebases the running total; the delta is deliberately dropped
	s_last_vu_time = THREAD_VU1 ? vu1Thread.GetThreadHandle().GetCPUTime() : 0;
	s_last_ticks = GetCPUTicks();

	for (GSSWThreadStats& stat : s_gs_sw_threads)
		stat.last_cpu_time = stat.handle.GetCPUTime();

	// F7 fork (P1): the MTGS thread restarts its F7PROF window at its next update.
	s_f7_rebase.store(true, std::memory_order_release);
}

void PerformanceMetrics::Update(bool gs_register_write, bool fb_blit, bool is_skipping_present)
{
	if (!is_skipping_present)
	{
		const float frame_time = s_last_frame_time.GetTimeMillisecondsAndReset();
		s_minimum_frame_time_accumulator = (s_minimum_frame_time_accumulator == 0.0f) ? frame_time : std::min(s_minimum_frame_time_accumulator, frame_time);
		s_average_frame_time_accumulator += frame_time;
		s_maximum_frame_time_accumulator = std::max(s_maximum_frame_time_accumulator, frame_time);
		s_frame_time_history[s_frame_time_history_pos] = frame_time;
		s_frame_time_history_pos = (s_frame_time_history_pos + 1) % NUM_FRAME_TIME_SAMPLES;
		s_unskipped_frames_since_last_update++;
	}

	s_frames_since_last_update++;
	s_gs_privileged_register_writes_since_last_update += static_cast<u32>(gs_register_write);
	s_gs_framebuffer_blits_since_last_update += static_cast<u32>(fb_blit);
	s_frame_number++;

	const Common::Timer::Value now_ticks = Common::Timer::GetCurrentValue();
	const Common::Timer::Value ticks_diff = now_ticks - s_last_update_time.GetStartValue();
	const float time = Common::Timer::ConvertValueToSeconds(ticks_diff);
	if (time < UPDATE_INTERVAL)
		return;

	s_last_update_time.ResetTo(now_ticks);
	s_minimum_frame_time = std::exchange(s_minimum_frame_time_accumulator, 0.0f);
	s_average_frame_time = std::exchange(s_average_frame_time_accumulator, 0.0f) / static_cast<float>(s_unskipped_frames_since_last_update);
	s_maximum_frame_time = std::exchange(s_maximum_frame_time_accumulator, 0.0f);
	s_fps = static_cast<float>(s_frames_since_last_update) / time;
	s_avg_vps.UpdateAvgFPS(s_fps);
	s_average_gpu_time = s_accumulated_gpu_time / static_cast<float>(s_unskipped_frames_since_last_update);
	s_average_gpu_vs_invocations = static_cast<double>(s_accumulated_gpu_vs_invocations) / static_cast<double>(s_unskipped_frames_since_last_update);
	s_average_gpu_ps_invocations = static_cast<double>(s_accumulated_gpu_ps_invocations) / static_cast<double>(s_unskipped_frames_since_last_update);
	s_gpu_usage = s_accumulated_gpu_time / (time * 10.0f);
	s_accumulated_gpu_time = 0.0f;
	s_accumulated_gpu_vs_invocations = 0;
	s_accumulated_gpu_ps_invocations = 0;

	// prefer privileged register write based framerate detection, it's less likely to have false positives
	if (s_gs_privileged_register_writes_since_last_update > 0 && !EmuConfig.Gamefixes.BlitInternalFPSHack)
	{
		s_internal_fps = static_cast<float>(s_gs_privileged_register_writes_since_last_update) / time;
		s_internal_fps_method = InternalFPSMethod::GSPrivilegedRegister;
	}
	else if (s_gs_framebuffer_blits_since_last_update > 0)
	{
		s_internal_fps = static_cast<float>(s_gs_framebuffer_blits_since_last_update) / time;
		s_internal_fps_method = InternalFPSMethod::DISPFBBlit;
	}
	else
	{
		s_internal_fps = 0;
		s_internal_fps_method = InternalFPSMethod::None;
	}

	s_gs_privileged_register_writes_since_last_update = 0;
	s_gs_framebuffer_blits_since_last_update = 0;

	const u64 ticks = GetCPUTicks();
	const u64 ticks_delta = ticks - s_last_ticks;
	s_last_ticks = ticks;

	const double pct_divider =
		100.0 * (1.0 / ((static_cast<double>(ticks_delta) * static_cast<double>(Threading::GetThreadTicksPerSecond())) /
						   static_cast<double>(GetTickFrequency())));
	const double time_divider = 1000.0 * (1.0 / static_cast<double>(Threading::GetThreadTicksPerSecond())) *
								(1.0 / static_cast<double>(s_frames_since_last_update));

	const u64 cpu_time = s_cpu_thread_handle.GetCPUTime();
	const u64 gs_time = MTGS::GetThreadHandle().GetCPUTime();
	const u64 gs_back_delta = SampleGSBackThreadCPUTime();
	const u64 vu_time = THREAD_VU1 ? vu1Thread.GetThreadHandle().GetCPUTime() : 0;

	const u64 cpu_delta = cpu_time - s_last_cpu_time;
	const u64 gs_delta = gs_time - s_last_gs_time;
	const u64 vu_delta = vu_time - s_last_vu_time;
	s_last_cpu_time = cpu_time;
	s_last_gs_time = gs_time;
	s_last_vu_time = vu_time;

	s_cpu_thread_usage = static_cast<double>(cpu_delta) * pct_divider;
	s_gs_thread_usage = static_cast<double>(gs_delta) * pct_divider;
	s_gs_back_thread_usage = static_cast<double>(gs_back_delta) * pct_divider;
	s_vu_thread_usage = static_cast<double>(vu_delta) * pct_divider;
	s_cpu_thread_time = static_cast<double>(cpu_delta) * time_divider;
	s_gs_thread_time = static_cast<double>(gs_delta) * time_divider;
	s_gs_back_thread_time = static_cast<double>(gs_back_delta) * time_divider;
	s_vu_thread_time = static_cast<double>(vu_delta) * time_divider;

	for (GSSWThreadStats& thread : s_gs_sw_threads)
	{
		const u64 time = thread.handle.GetCPUTime();
		const u64 delta = time - thread.last_cpu_time;
		thread.last_cpu_time = time;
		thread.usage = static_cast<double>(delta) * pct_divider;
		thread.time = static_cast<double>(delta) * time_divider;
	}

	// F7 fork (P1): feed this update's window to the F7PROF line.
	F7ProfOnUpdate(time, s_frames_since_last_update, cpu_delta, gs_delta, gs_back_delta, vu_delta);

	// Rolling perf log (uses this window's frame count before it resets).
	s_log_accum_time += time;
	s_log_accum_frames += s_frames_since_last_update;
	if (s_log_accum_time >= LOG_INTERVAL)
	{
		// The back thread only exists with GS multi-threading on, so the field is
		// omitted rather than logged as a permanent 0% in the default configuration.
		char gs_back[32] = {};
		if (HasGSBackThread())
			std::snprintf(gs_back, sizeof(gs_back), " GSB %.0f%%", s_gs_back_thread_usage);

		Console.WriteLn("PerfLog: %.1f fps | EE %.0f%% GS %.0f%%%s VU %.0f%% GPU %.0f%% | frame %llu",
			static_cast<float>(s_log_accum_frames) / s_log_accum_time, s_cpu_thread_usage,
			s_gs_thread_usage, gs_back, s_vu_thread_usage, s_gpu_usage,
			static_cast<unsigned long long>(s_frame_number));
		s_log_accum_time = 0.0f;
		s_log_accum_frames = 0;
	}

	s_frames_since_last_update = 0;
	s_unskipped_frames_since_last_update = 0;
	s_presents_since_last_update = 0;

	Host::OnPerformanceMetricsUpdated();
}

void PerformanceMetrics::OnGPUPresent(float gpu_time, u64 vs_invocations, u64 ps_invocations)
{
	s_last_gpu_time = gpu_time;
	s_accumulated_gpu_time += gpu_time;
	s_accumulated_gpu_vs_invocations += vs_invocations;
	s_accumulated_gpu_ps_invocations += ps_invocations;
	s_presents_since_last_update++;
}

void PerformanceMetrics::SetCPUThread(Threading::ThreadHandle thread)
{
	s_last_cpu_time = thread ? thread.GetCPUTime() : 0;
	s_cpu_thread_handle = std::move(thread);
}

void PerformanceMetrics::AdpfRegisterCallingThread()
{
#if defined(__ANDROID__)
	const int32_t tid = static_cast<int32_t>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	if (std::find(s_adpf_tids.begin(), s_adpf_tids.end(), tid) != s_adpf_tids.end())
		return;
	s_adpf_tids.push_back(tid);
	// The session's thread list is fixed at creation, so a newly-registered thread
	// means the current session is missing it — drop it and let the next frame
	// recreate it over the full set.
	if (s_adpf_session)
	{
		s_adpf.closeSession(s_adpf_session);
		s_adpf_session = nullptr;
	}
	s_adpf_create_failed = false;
#endif
}

void PerformanceMetrics::AdpfSetEnabled(bool enabled)
{
#if defined(__ANDROID__)
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	if (s_adpf_enabled == enabled)
		return;
	s_adpf_enabled = enabled;
	if (!enabled && s_adpf_session)
	{
		s_adpf.closeSession(s_adpf_session);
		s_adpf_session = nullptr;
	}
	if (enabled)
		s_adpf_create_failed = false; // allow the next frame to recreate
#else
	(void)enabled;
#endif
}

void PerformanceMetrics::AdpfShutdown()
{
#if defined(__ANDROID__)
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	if (s_adpf_session)
	{
		s_adpf.closeSession(s_adpf_session);
		s_adpf_session = nullptr;
	}
	s_adpf_tids.clear();
	s_adpf_create_failed = false;
	s_adpf_work_start = 0;
	s_adpf_work_accum_ns = 0;
#endif
}

void PerformanceMetrics::AdpfEndWorkSegment(bool frame_end)
{
#if defined(__ANDROID__)
	// Sampled at Throttle() entry — the instant the frame's active CPU work finished, before the
	// limiter sleep — so (now - work_start) excludes the deliberate limiter sleep. It is NOT pure
	// CPU compute: the EE can still block behind a full MTGS queue that is itself stalled on
	// presentation, so some present-wait can leak in. Acceptable for a first experiment, and a far
	// better approximation of ADPF's "last workload cycle" than the present interval.
	const Common::Timer::Value now = Common::Timer::GetCurrentValue();
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	if (!s_adpf_enabled)
		return;
	AdpfEnsureSession();
	if (!s_adpf_session || s_adpf_work_start == 0)
	{
		// No open segment (first frame, or just resumed from a pause). If this is the frame's
		// first wait, its second segment alone would under-report the frame, so skip it.
		s_adpf_work_accum_ns = frame_end ? 0 : -1;
		return;
	}
	const int64_t segment_ns = static_cast<int64_t>(Common::Timer::ConvertValueToSeconds(now - s_adpf_work_start) * 1.0e9);
	if (!frame_end)
	{
		if (s_adpf_work_accum_ns >= 0)
			s_adpf_work_accum_ns += segment_ns;
		return;
	}
	if (s_adpf_work_accum_ns < 0)
	{
		s_adpf_work_accum_ns = 0;
		return;
	}
	const int64_t work_ns = s_adpf_work_accum_ns + segment_ns;
	s_adpf_work_accum_ns = 0;
	const int64_t target = AdpfTargetNs();
	if (target > 0 && target != s_adpf_target_ns)
	{
		s_adpf.updateTarget(s_adpf_session, target);
		s_adpf_target_ns = target;
	}
	// Drop absurd outliers (savestate load, renderer recreation, debugger stall): a period several
	// times the deadline is not a real frame and would spam a spurious max-frequency demand.
	if (work_ns <= 0 || (s_adpf_target_ns > 0 && work_ns > s_adpf_target_ns * 4))
		return;
	if (s_adpf_paused)
	{
		Console.WriteLn("ADPF: reporting resumed.");
		s_adpf_paused = false;
	}
	const int ret = s_adpf.reportActual(s_adpf_session, work_ns);
	if (ret != 0 && !s_adpf_report_warned)
	{
		s_adpf_report_warned = true;
		Console.Warning("ADPF: reportActualWorkDuration returned %d — driver is ignoring the hint.", ret);
	}
#endif
}

void PerformanceMetrics::AdpfBeginFrameWork()
{
#if defined(__ANDROID__)
	// Opens a work period at the post-sleep instant (Throttle exit), so the deliberate limiter
	// sleep is excluded from the next reported duration.
	const Common::Timer::Value now = Common::Timer::GetCurrentValue();
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	s_adpf_work_start = now;
#endif
}

void PerformanceMetrics::AdpfPauseFrameWork()
{
#if defined(__ANDROID__)
	// Not frame-limiting (unlimited / host-vsync / interrupted) — invalidate the period so no
	// wall-time-with-wait duration is reported, and edge-log so a tester never sees "ACTIVE" while
	// nothing is actually being submitted.
	std::lock_guard<std::mutex> lock(s_adpf_mutex);
	if (s_adpf_session && !s_adpf_paused)
	{
		Console.WriteLn("ADPF: reporting paused (unlimited / host-vsync / interrupted) — no durations submitted.");
		s_adpf_paused = true;
	}
	s_adpf_work_start = 0;
	s_adpf_work_accum_ns = 0;
#endif
}

void PerformanceMetrics::SetGSSWThreadCount(u32 count)
{
	s_gs_sw_threads.clear();
	s_gs_sw_threads.resize(count);
}

void PerformanceMetrics::SetGSSWThread(u32 index, Threading::ThreadHandle thread)
{
	s_gs_sw_threads[index].last_cpu_time = thread ? thread.GetCPUTime() : 0;
	s_gs_sw_threads[index].handle = std::move(thread);
}

void PerformanceMetrics::SetGSBackThread(Threading::ThreadHandle thread)
{
	std::unique_lock lock(s_gs_back_thread_mutex);
	// Rebase off the incoming handle, so the first window after a GSreopen respawn measures
	// this thread's time rather than its difference against the retired thread's total.
	s_last_gs_back_time = thread ? thread.GetCPUTime() : 0;
	s_gs_back_thread_handle = std::move(thread);
}

u64 PerformanceMetrics::GetFrameNumber()
{
	return s_frame_number;
}

PerformanceMetrics::InternalFPSMethod PerformanceMetrics::GetInternalFPSMethod()
{
	return s_internal_fps_method;
}

bool PerformanceMetrics::IsInternalFPSValid()
{
	return s_internal_fps_method != InternalFPSMethod::None;
}

float PerformanceMetrics::GetFPS()
{
	return s_fps;
}

float PerformanceMetrics::GetAvgVPS()
{
	return s_avg_vps.GetAvgFPS();
}

float PerformanceMetrics::GetInternalFPS()
{
	return s_internal_fps;
}

float PerformanceMetrics::GetSpeed()
{
	return (s_fps / VMManager::GetFrameRate()) * 100.0;
}

float PerformanceMetrics::GetAverageFrameTime()
{
	return s_average_frame_time;
}

float PerformanceMetrics::GetMinimumFrameTime()
{
	return s_minimum_frame_time;
}

float PerformanceMetrics::GetMaximumFrameTime()
{
	return s_maximum_frame_time;
}

double PerformanceMetrics::GetCPUThreadUsage()
{
	return s_cpu_thread_usage;
}

double PerformanceMetrics::GetCPUThreadAverageTime()
{
	return s_cpu_thread_time;
}

float PerformanceMetrics::GetGSThreadUsage()
{
	return s_gs_thread_usage;
}

float PerformanceMetrics::GetGSThreadAverageTime()
{
	return s_gs_thread_time;
}

bool PerformanceMetrics::HasGSBackThread()
{
	std::unique_lock lock(s_gs_back_thread_mutex);
	return static_cast<bool>(s_gs_back_thread_handle);
}

float PerformanceMetrics::GetGSBackThreadUsage()
{
	return s_gs_back_thread_usage;
}

float PerformanceMetrics::GetGSBackThreadAverageTime()
{
	return s_gs_back_thread_time;
}

float PerformanceMetrics::GetVUThreadUsage()
{
	return s_vu_thread_usage;
}

float PerformanceMetrics::GetVUThreadAverageTime()
{
	return s_vu_thread_time;
}

u32 PerformanceMetrics::GetGSSWThreadCount()
{
	return static_cast<u32>(s_gs_sw_threads.size());
}

double PerformanceMetrics::GetGSSWThreadUsage(u32 index)
{
	return s_gs_sw_threads[index].usage;
}

double PerformanceMetrics::GetGSSWThreadAverageTime(u32 index)
{
	return s_gs_sw_threads[index].time;
}

float PerformanceMetrics::GetGPUUsage()
{
	return s_gpu_usage;
}

float PerformanceMetrics::GetGPUAverageTime()
{
	return s_average_gpu_time;
}

float PerformanceMetrics::GetLastGPUTime()
{
	return s_last_gpu_time;
}

double PerformanceMetrics::GetGPUAverageVSInvocations()
{
	return s_average_gpu_vs_invocations;
}

double PerformanceMetrics::GetGPUAveragePSInvocations()
{
	return s_average_gpu_ps_invocations;
}

const PerformanceMetrics::FrameTimeHistory& PerformanceMetrics::GetFrameTimeHistory()
{
	return s_frame_time_history;
}

u32 PerformanceMetrics::GetFrameTimeHistoryPos()
{
	return s_frame_time_history_pos;
}
