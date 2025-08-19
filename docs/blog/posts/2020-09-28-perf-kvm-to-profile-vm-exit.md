---
title: perf kvm to profile vm_exit
date: 2020-09-28 18:20:47
thumbnail: "/images/music-equalizer-1.jpg"
description: "Profiling VM_EXIT reasons with `perf kvm` tool"
tags:
 - linux
 - perf
---

Optimizing VM_EXITs will significantly improve performance VMs. All the major improvements in VM world is mainly focusing on reducing the number of VM_EXITs. To optimize it, first we should able to measure it. Initially the tool `kvm_stat` was designed for this purpose, later it has been added inside `perf` itself.

To profile VM_EXITs while running `sysbench`,
 * Get `pid` of the VM task - 127894
 * Get the IP of that machine - 192.168.122.194
     Make sure you can ssh to that machine without password
 * Install `sysbench` inside the VM

```sh
$ sudo perf kvm stat record -p 127894 ssh 192.168.122.194 -l test_user "sysbench --test=cpu --cpu-max-prime=20000 run"
sysbench 0.4.12:  multi-threaded system evaluation benchmark

Running the test with following options:
Number of threads: 1

Doing CPU performance benchmark

Threads started!
Done.

Maximum prime number checked in CPU test: 20000


Test execution summary:
    total time:                          22.6607s
    total number of events:              10000
    total time taken by event execution: 22.6598
    per-request statistics:
         min:                                  2.13ms
         avg:                                  2.27ms
         max:                                 12.10ms
         approx.  95 percentile:               2.88ms

Threads fairness:
    events (avg/stddev):           10000.0000/0.00
    execution time (avg/stddev):   22.6598/0.00

[ perf record: Woken up 2 times to write data ]
[ perf record: Captured and wrote 4.779 MB perf.data.guest (52461 samples) ]
$
```

Perf has recorded the data in `perf.data.guest` in the current directory. Now to view VM_EXITs,

```sh
$ sudo perf kvm stat report --event=vmexit


Analyze events for all VMs, all VCPUs:

             VM-EXIT    Samples  Samples%     Time%    Min Time    Max Time         Avg time

           MSR_WRITE       9167    35.40%     0.04%      0.45us   9554.94us      3.00us ( +-  41.94% )
  EXTERNAL_INTERRUPT       5877    22.69%     0.02%      0.37us   1175.48us      2.43us ( +-  17.90% )
    PREEMPTION_TIMER       5728    22.12%     0.01%      0.51us     21.14us      0.62us ( +-   0.87% )
                 HLT       2232     8.62%    99.92%      0.56us 1001118.99us  30567.94us ( +-   9.88% )
               CPUID       2160     8.34%     0.00%      0.40us     12.82us      0.65us ( +-   1.29% )
   PAUSE_INSTRUCTION        390     1.51%     0.00%      0.38us   1490.19us      8.27us ( +-  62.22% )
       EPT_MISCONFIG        303     1.17%     0.01%      1.04us    167.13us     13.33us ( +-   8.61% )
         EOI_INDUCED         37     0.14%     0.00%      0.62us      3.00us      1.24us ( +-   6.58% )
       EXCEPTION_NMI          4     0.02%     0.00%      0.42us      0.56us      0.47us ( +-   6.81% )

Total Samples:25898, Total events handled time:68281638.61us.

$
```
