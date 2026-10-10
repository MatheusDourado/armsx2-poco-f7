// SPDX-FileCopyrightText: 2026 ARMSX2 Poco F7 fork
// SPDX-License-Identifier: GPL-3.0+

#pragma once

// Poco F7 (SM8735, Adreno 825, 12 GB) fork additions that more than one file needs. They are kept
// in one header so the fork's diff against upstream stays small and easy to rebase:
//   F7Prof   (P1) always-on wait counters for the GS front (MTGS) and GS Back threads, printed
//                 every ~5 s as one console line starting with "F7PROF" (PerformanceMetrics.cpp).
//   F7Tuning (P6) the 12 GB RAM gate for the larger GS texture pool and hash cache.
// The prime-core placement (P2, affinity mode 8) lives entirely in VMManager.cpp.

#include "common/HostSys.h"
#include "common/Pcsx2Defs.h"

#include <atomic>
#include <cstddef>

namespace F7Prof
{
	/// One kind of wait: total time in GetCPUTicks() units, and how many waits. Every counter has
	/// a single writer thread, so an update is a relaxed load plus a relaxed store (plain LDR/STR,
	/// no read-modify-write, no lock). The reporter on the MTGS thread reads them with relaxed
	/// loads and only ever uses the difference between two reads.
	struct Counter
	{
		std::atomic<u64> ticks{0};
		std::atomic<u64> count{0};

		void Add(u64 dt)
		{
			ticks.store(ticks.load(std::memory_order_relaxed) + dt, std::memory_order_relaxed);
			count.store(count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		}
	};

	/// Single-writer event count (no time).
	inline void Bump(std::atomic<u64>& counter, u64 n = 1)
	{
		counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
	}

	/// Why the MTGS thread waited for the GS back thread to go idle (GSState::DrainBackQueue).
	enum class Drain : u8
	{
		Other, ///< settings, reset, savestates, screenshots, OSD/UI, texture-cache purges, ...
		Vsync, ///< once per field: GSRenderer::SubmitVsync, before Merge/present
		Fifo, ///< EE FIFO readback (GSState::InitReadFIFO / GSState::Read)
		AA1, ///< AA1 coverage query (GSFrontState::IsCoverageAlphaSupported)
		Move, ///< local->local move while a GameDB move hook is armed
		Count
	};

	/// Which producer resource ran out when the MTGS thread blocked (GSBackQueue::SpaceWait).
	enum class Full : u8
	{
		Ring, ///< record ring (512 slots)
		DrawPool, ///< draw vertex/index nodes (64)
		PayloadPool, ///< 4 MB transfer payload nodes (8)
		Count
	};

	/// Written only by the MTGS (GS front) thread.
	struct FrontCounters
	{
		Counter vu; ///< waiting for MTVU to finish a VU1 program (semaXGkick), lock re-take included
		std::atomic<u64> vu_packets{0}; ///< every VU1 xgkick packet, waited for or not
		Counter drain[static_cast<size_t>(Drain::Count)];
		Counter full[static_cast<size_t>(Full::Count)];
		Counter idle; ///< MTGS main loop asleep, waiting for the EE to queue work
		Counter vsync_body; ///< after the vsync drain: Merge, present, OSD (GSRenderer::SubmitVsync)
		Counter gpu; ///< vkWaitForFences (and the readback spin) issued from this thread
	};

	/// Written only by the GS Back thread.
	struct BackCounters
	{
		Counter idle; ///< waiting for records, including the up-to-50us spin before it sleeps
		Counter gpu; ///< vkWaitForFences (and the readback spin) issued from this thread
		std::atomic<u64> records{0}; ///< records executed
	};

	inline FrontCounters g_front;
	inline BackCounters g_back;

	/// The calling thread's GPU-wait counter. The MTGS and GS Back threads bind theirs when they
	/// start; it stays null on every other thread, whose waits are not counted.
	inline thread_local Counter* t_gpu = nullptr;

	inline u64 Now()
	{
		return GetCPUTicks();
	}

	inline void BindGpuCounter(Counter* counter)
	{
		t_gpu = counter;
	}

	inline void AddGpuWait(u64 dt)
	{
		if (Counter* counter = t_gpu)
			counter->Add(dt);
	}

	inline void AddDrain(Drain why, u64 dt)
	{
		g_front.drain[static_cast<size_t>(why)].Add(dt);
	}

	inline void AddFull(Full why, u64 dt)
	{
		g_front.full[static_cast<size_t>(why)].Add(dt);
	}
} // namespace F7Prof

namespace F7Tuning
{
	/// 12 GB phones report about 10.9-11.2 GiB through sysconf (the kernel and firmware carve-outs
	/// keep the rest; the Poco F7 reads 10.97 GiB), 8 GB ones about 7.2-7.6 GiB, so 10 GiB tells
	/// the two classes apart. Swap (HyperOS "memory extension") is not physical memory and is not
	/// counted by sysconf either.
	inline constexpr u64 BIG_RAM_BYTES = 10ull * 1024 * 1024 * 1024;

	/// GSTextureCache::AgeHashCache limits on such devices (upstream: 800 entries, 30 frames).
	inline constexpr u32 HASH_CACHE_SIZE = 1600;
	inline constexpr u32 HASH_CACHE_AGE = 60;

	inline bool HasBigRam()
	{
		static const bool big = (GetPhysicalMemory() >= BIG_RAM_BYTES);
		return big;
	}
} // namespace F7Tuning
