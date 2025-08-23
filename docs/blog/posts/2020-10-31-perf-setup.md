---
title: perf setup
date: 2020-10-31 21:17:02
thumbnail: "/images/batmobile-tank.jpg"
description: This post describes the necessary setup for Linux-perf. The setup machine is an Ubuntu VM. No custom kernel build.
tags:
 - linux
 - ubuntu
 - kernel
 - perf
 - setup
---

`Linux-perf` aka `perf` is versatile, like a batmobile. It has all the tools and functionalities you need. And you'll feel like a superhero once you master it.

By default perf comes with may tools that relying on debug and trace symbols exported via `procfs`. But to add custom probes and probes with line numbers, kernel debug symbols and kernel source is necessary. In this post I'll walk you through the necessary setup process. I'm using an Ubuntu-20.04 VM running on Virtual box. I'm not going to rebuild and install kernel. The steps will be,

 * Install Linux-kernel debug symbols
 * Fetch Linux-kernel source
 * Install perf
 * First run

## Enable non-common repositories

Enable debug repositories in `apt` source list.

```sh
bala@ubuntu-vm-1:~$ echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list.d/ddebs.list
```

Install debug keyring
```sh
bala@ubuntu-vm-1:~$ sudo apt install ubuntu-dbgsym-keyring
```

Enable source repositories in `apt` source list.
```sh
bala@ubuntu-vm-1:~$ grep deb-src /etc/apt/sources.list
deb-src http://in.archive.ubuntu.com/ubuntu/ focal main restricted
```

Do `apt` update
```sh
bala@ubuntu-vm-1:~$ sudo apt update
```

## 1. Install Linux-kernel debug symbols
Install Linux debug symbols corresponding the kernel installed in your machine.

```sh
bala@ubuntu-vm-1:~$ sudo apt install -y linux-image-`uname -r`-dbgsym
```

Linux image with debug symbols will be installed in the directory `/usr/lib/debug/boot/`
```sh
bala@ubuntu-vm-1:~$ ls -lh /usr/lib/debug/boot/vmlinux-5.4.0-52-generic
-rw-r--r-- 2 root root 742M Oct 15 15:58 /usr/lib/debug/boot/vmlinux-5.4.0-52-generic
bala@ubuntu-vm-1:~$
```

## 2. Fetch Linux kernel source
Fetch the source package corresponding to the installed kernel.
```sh
bala@ubuntu-vm-1:~$ sudo apt install linux-source
```
Kernel source with debian packaging files will be installed in the path `/usr/src/linux-source-5.4.0`. The kernel source is available in a tarball inside this directory. Copy to your desired location and extract.
```sh
bala@ubuntu-vm-1:~$ ls -lh /usr/src/linux-source-5.4.0/linux-source-5.4.0.tar.bz2
-rw-r--r-- 1 root root 129M Oct 15 15:58 /usr/src/linux-source-5.4.0/linux-source-5.4.0.tar.bz2
bala@ubuntu-vm-1:~$ cp -f /usr/src/linux-source-5.4.0/linux-source-5.4.0.tar.bz2 ~/source/
bala@ubuntu-vm-1:~$ cd ~/source/
bala@ubuntu-vm-1:~/source$ tar -xvf linux-source-5.4.0.tar.bz2
bala@ubuntu-vm-1:~/source$ ls ~/source/linux-source-5.4.0/
arch   certs    CREDITS  Documentation  dropped.txt  include  ipc     Kconfig  lib       MAINTAINERS  mm   README   scripts   snapcraft.yaml  tools   update-version-dkms  virt
block  COPYING  crypto   drivers        fs           init     Kbuild  kernel   LICENSES  Makefile     net  samples  security  sound           ubuntu  usr
bala@ubuntu-vm-1:~$
```
## 3. Install Linux perf
It comes with `linux-tools-generic` package on Ubuntu-20.04.
```
bala@ubuntu-vm-1:~$ sudo apt install linux-tools-generic
```

## 4. Run your first perf command
I want to count number of IPI (Inter Processor Interrupts) sent by `resched_curr`. It sends IPI when the target CPU is not the current CPU (the one executing the function itself). Here is the source code of that function.
```C
void resched_curr(struct rq *rq)
{
	struct task_struct *curr = rq->curr;
	int cpu;

	lockdep_assert_held(&rq->lock);

	if (test_tsk_need_resched(curr))
		return;

	cpu = cpu_of(rq);

	if (cpu == smp_processor_id()) {
		set_tsk_need_resched(curr);
		set_preempt_need_resched();
		return;
	}

	if (set_nr_and_not_polling(curr))
		smp_send_reschedule(cpu);
	else
		trace_sched_wake_idle_without_ipi(cpu);
}
```

So if target CPU is the current CPU, line number 14 will get executed. Otherwise execution continues from line number 18. Also I want to record the target CPU in both cases.

**Get the line numbers where you can insert probes from perf itself.**
```sh
bala@ubuntu-vm-1:~/source$ sudo perf probe -k /usr/lib/debug/boot/vmlinux-5.4.0-52-generic -s ~/source/linux-source-5.4.0 -L resched_curr
<resched_curr@/home/bala/source/linux-source-5.4.0//kernel/sched/core.c:0>
      0  void resched_curr(struct rq *rq)
      1  {
      2         struct task_struct *curr = rq->curr;
                int cpu;

                lockdep_assert_held(&rq->lock);

      7         if (test_tsk_need_resched(curr))
                        return;

     10         cpu = cpu_of(rq);

     12         if (cpu == smp_processor_id()) {
     13                 set_tsk_need_resched(curr);
     14                 set_preempt_need_resched();
     15                 return;
                }

     18         if (set_nr_and_not_polling(curr))
     19                 smp_send_reschedule(cpu);
                else
     21                 trace_sched_wake_idle_without_ipi(cpu);
         }

         void resched_cpu(int cpu)

bala@ubuntu-vm-1:~/source$

```

Here is the probe for non-IPI case. I name it as `resched_curr_same_cpu`.
```sh
bala@ubuntu-vm-1:~$ sudo perf probe -k /usr/lib/debug/boot/vmlinux-5.4.0-52-generic -s source/linux-source-5.4.0 resched_curr_same_cpu='resched_curr:14 rq->cpu'
```

Probe for IPI case. And I name it as `resched_curr_send_ipi`.
```sh
bala@ubuntu-vm-1:~$ sudo perf probe -k /usr/lib/debug/boot/vmlinux-5.4.0-52-generic -s source/linux-source-5.4.0 resched_curr_send_ipi='resched_curr:19 rq->cpu'
```

*Note: To probe the function `resched_curr` and its argument `rq`, we need Linux debug symbols. And to probe on line numbers we need Linux source. So that we have installed both of them earlier.*

Now lets capture the execution of a `stress-ng` test.
```sh
bala@ubuntu-vm-1:~$ sudo perf record -e probe:resched_curr_same_cpu,probe:resched_curr_send_ipi stress-ng --mq 8 -t 5 --metrics-brief
stress-ng: info:  [22439] dispatching hogs: 8 mq
stress-ng: info:  [22439] successful run completed in 5.01s
stress-ng: info:  [22439] stressor       bogo ops real time  usr time  sys time   bogo ops/s   bogo ops/s
stress-ng: info:  [22439]                           (secs)    (secs)    (secs)   (real time) (usr+sys time)
stress-ng: info:  [22439] mq              2225397      5.00      3.57     16.14    445062.30    112907.00
[ perf record: Woken up 421 times to write data ]
[ perf record: Captured and wrote 105.404 MB perf.data (1380709 samples) ]
bala@ubuntu-vm-1:~$
```

And the report is,
```sh
bala@ubuntu-vm-1:~$ sudo perf report --stdio
# To display the perf.data header info, please use --header/--header-only options.
#
#
# Total Lost Samples: 0
#
# Samples: 1M of event 'probe:resched_curr_same_cpu'
# Event count (approx.): 1380698
#
# Overhead  Trace output
# ........  ........................
#
    29.13%  (ffffffff83ad740d) cpu=1
    27.77%  (ffffffff83ad740d) cpu=2
    24.74%  (ffffffff83ad740d) cpu=0
    18.36%  (ffffffff83ad740d) cpu=3


# Samples: 11  of event 'probe:resched_curr_send_ipi'
# Event count (approx.): 11
#
# Overhead  Trace output
# ........  ........................
#
    45.45%  (ffffffff83ad73af) cpu=1
    36.36%  (ffffffff83ad73af) cpu=3
     9.09%  (ffffffff83ad73af) cpu=0
     9.09%  (ffffffff83ad73af) cpu=2


#
# (Cannot load tips.txt file, please install perf!)
#
bala@ubuntu-vm-1:~$
```
As you can see only 11 times out of a million times an IPI is sent. More on this in later posts. Until then... _"Perhaps you should read the instructions first?"_.

# References
 * http://www.brendangregg.com/perf.html
 * https://wiki.ubuntu.com/Kernel/Reference/stress-ng
 * https://man7.org/linux/man-pages/man1/perf-probe.1.html
 * https://wiki.ubuntu.com/Debug%20Symbol%20Packages
 * https://askubuntu.com/questions/50145/how-to-install-perf-monitoring-tool
