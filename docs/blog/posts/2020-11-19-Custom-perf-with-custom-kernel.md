---
title: Custom perf with custom kernel
date: 2020-11-19 18:46:42
thumbnail: "/images/batmobile-tank.jpg"
tags:
 - linux
 - kernel
 - perf
---

There is no doubt that perf is an awesome tool. And eBPF enables us to attach an arbitrary code to any tracepoint. But both has one limitation. They cannot execute a kernel function. As if I want to call a kernel function whenever a tracepoint is hit. Its not possible today. It is not a big caveat. But will be very much useful while debugging kernel.

Lets say I want to trace all IPIs and want to record the source CPU, target CPU. And I want to know whether Idle task was running on the target CPU during the IPI. Inter Processor Interrupt or IPI is and interrupt sent from one CPU to another to wake it up.

**NOTE**: *All the code referred here is from Linux-5.4.0*

The code that sends IPI in x86 platform is `native_smp_send_reschedule`. Below is the code of that function.
```C
void native_smp_send_reschedule(int cpu)
{
        if (unlikely(cpu_is_offline(cpu))) {
                WARN(1, "sched: Unexpected reschedule of offline CPU#%d!\n", cpu);
                return;
        }
        apic->send_IPI(cpu, RESCHEDULE_VECTOR);
}
```
The only argument of this function `cpu` mentions the target CPU - to which CPU IPI is being sent. I want to source CPU from which the IPI is originates.

To get current CPU kernel has a macro - `smp_processor_id()`.

We can get the task-run-queue of that CPU using the macro `cpu_rq(int cpu)`. And `cpu_rq(int cpu)->curr` will point to the task that is currently running on that CPU. If that task's `pid` is 0, it is the Idle task.

So my deduced requirement will be setting and tracepoint on `native_smp_send_reschedule` and call these kernel functions upon hitting that tracepoint. There is no way today. Thus I'm left with only one option - build my own kernel with necessary modifications.

## Building custom kernel
I've cloned kernel repository from kernel.org. And checked-out to the desired branch. I'm running Ubuntu-20.05 inside a Virtualbox. So I'm using the kernel-5.4.0 which is already installed in my distribution.
```sh
bala@ubuntu-vm-1:~/source/$ git clone git://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
bala@ubuntu-vm-1:~/source/$ cd linux
bala@ubuntu-vm-1:~/source/linux/$ git checkout -b v5.4.0 v5.4
bala@ubuntu-vm-1:~/source/linux/$ head -n6 Makefile
 # SPDX-License-Identifier: GPL-2.0
 VERSION = 5
 PATCHLEVEL = 4
 SUBLEVEL = 0
 EXTRAVERSION =
 NAME = Kleptomaniac Octopus
bala@ubuntu-vm-1:~/source/linux/$
```

Copied config from current OS. And `make oldconfig`
```sh
bala@ubuntu-vm-1:~/source/linux/$ cp /boot/config-5.4.0-52-generic ./.config
bala@ubuntu-vm-1:~/source/linux/$ make oldconfig
```
If any new configs were, select appropriately if you know it. Otherwise simply ignore it.

Applied following patch.
```sh
bala@ubuntu-vm-1:~/source/linux$ git diff
diff --git a/arch/x86/kernel/apic/ipi.c b/arch/x86/kernel/apic/ipi.c
index 6ca0f91372fd..31df1e1f4922 100644
--- a/arch/x86/kernel/apic/ipi.c
+++ b/arch/x86/kernel/apic/ipi.c
@@ -2,6 +2,7 @@

 #include <linux/cpumask.h>
 #include <linux/smp.h>
+#include "../../../../kernel/sched/sched.h"

 #include "local.h"

@@ -61,14 +62,22 @@ void apic_send_IPI_allbutself(unsigned int vector)
  * wastes no time serializing anything. Worst case is that we lose a
  * reschedule ...
  */
+#pragma GCC push_options
+#pragma GCC optimize ("O0")
 void native_smp_send_reschedule(int cpu)
 {
+       int this_cpu = smp_processor_id();
+       int that_cpu = cpu;
+       struct rq *rq = cpu_rq(that_cpu);
+       int is_idle_task = ((rcu_dereference(rq->curr))->pid == 0);
+
        if (unlikely(cpu_is_offline(cpu))) {
                WARN(1, "sched: Unexpected reschedule of offline CPU#%d!\n", cpu);
                return;
        }
        apic->send_IPI(cpu, RESCHEDULE_VECTOR);
 }
+#pragma GCC pop_options

 void native_send_call_func_single_ipi(int cpu)
 {
diff --git a/lib/Kconfig.debug b/lib/Kconfig.debug
index 93d97f9b0157..706fdfc715ab 100644
--- a/lib/Kconfig.debug
+++ b/lib/Kconfig.debug
@@ -359,6 +359,7 @@ config SECTION_MISMATCH_WARN_ONLY
 #
 config ARCH_WANT_FRAME_POINTERS
        bool
+       default y

 config FRAME_POINTER
        bool "Compile the kernel with frame pointers"
bala@ubuntu-vm-1:~/source/linux$
```
This patch contains two changes,
 * `lib/Kconfig.debug` - It is to enable frame pointers. Frame pointers will be very much useful for stack trace.
 * `arch/x86/kernel/apic/ipi.c`
     * `pragma` instructions are compiler directives. They tell GCC not to optimize the code below - `O0` optimization. Otherwise, GCC may optimize away the variables as they are unused variables.
     * `this_cpu` is got from `smp_processor_id()`
     * `is_idle_task` will be set to 1 if target CPU is executing Idle task.
     * Last `pragma` instruction is to reset GCC options back to default.

Enabled kernel debug info by setting `CONFIG_DEBUG_INFO` in the `.config` file. And built the kernel. My VM has 4 CPUs, so started 8 parallel build threads.
```sh
bala@ubuntu-vm-1:~/source/linux$ make bzImage -j8
...
...
...
  OBJCOPY arch/x86/boot/setup.bin
  BUILD   arch/x86/boot/bzImage
Setup is 16412 bytes (padded to 16896 bytes).
System is 8097 kB
CRC 65acc728
Kernel: arch/x86/boot/bzImage is ready  (#1)
bala@ubuntu-vm-1:~/source/linux$
```

Now copied the `bzImage` to `/boot/` directory and updated boot loader.
```sh
bala@ubuntu-vm-1:~/source/linux$ sudo cp arch/x86/boot/bzImage /boot/vmlinuz-dbg-custom
bala@ubuntu-vm-1:~/source/linux$ sudo update-grub
Sourcing file `/etc/default/grub'
Sourcing file `/etc/default/grub.d/init-select.cfg'
Generating grub configuration file ...
dpkg: warning: version 'dbg-custom' has bad syntax: version number does not start with digit
Found linux image: /boot/vmlinuz-dbg-custom
Found linux image: /boot/vmlinuz-5.4.0-52-generic
Found initrd image: /boot/initrd.img-5.4.0-52-generic
done
bala@ubuntu-vm-1:~/source/linux$
```

Rebooted the VM into newly built kernel.

## Custom build perf
As the kernel is custom built, the perf package comes with Ubuntu-20.04 didn't get executed. It threw following error.
```sh
bala@ubuntu-vm-1:~/source/linux$ perf --help
WARNING: perf not found for kernel 5.4.0+

  You may need to install the following packages for this specific kernel:
    linux-tools-5.4.0+-5.4.0+
    linux-cloud-tools-5.4.0+-5.4.0+

  You may also want to install one of the following packages to keep up to date:
    linux-tools-5.4.0+
    linux-cloud-tools-5.4.0+
bala@ubuntu-vm-1:~/source/linux$
```

So I went ahead and build perf from kernel source itself. Perf requires following packages for adding a probe and source line probing.
 * libelf-dev
 * libdw-dev

I installed both of them.
```sh
bala@ubuntu-vm-1:~/source/linux/$ sudo apt install libelf-dev libdw-dev
```

Now built perf inside the kernel source tree.
```sh
bala@ubuntu-vm-1:~/source/linux/$ cd tools/perf
bala@ubuntu-vm-1:~/source/linux/tools/perf/$ make
```

## Running the probe on custom kernel with custom perf
```sh
bala@ubuntu-vm-1:~/source/linux/tools/perf$ sudo ./perf probe -s /home/bala/source/linux/ -k ../../vmlinux native_smp_send_reschedule="native_smp_send_reschedule:7 this_cpu that_cpu is_idle_task"
Added new events:
  probe:native_smp_send_reschedule (on native_smp_send_reschedule:7 with this_cpu that_cpu is_idle_task)
  probe:native_smp_send_reschedule_1 (on native_smp_send_reschedule:7 with this_cpu that_cpu is_idle_task)

You can now use it in all perf tools, such as:

        perf record -e probe:native_smp_send_reschedule_1 -aR sleep 1

bala@ubuntu-vm-1:~/source/linux/tools/perf$ sudo ./perf record -e probe:native_smp_send_reschedule_1 -aR sleep 1
Couldn't synthesize bpf events.
[ perf record: Woken up 1 times to write data ]
way too many cpu caches..[ perf record: Captured and wrote 0.101 MB perf.data (10 samples) ]
bala@ubuntu-vm-1:~/source/linux/tools/perf$ sudo ./perf report --stdio
# To display the perf.data header info, please use --header/--header-only options.
#
#
# Total Lost Samples: 0
#
# Samples: 10  of event 'probe:native_smp_send_reschedule_1'
# Event count (approx.): 10
#
# Overhead  Trace output
# ........  .......................................................
#
    20.00%  (ffffffffa5e46bc9) this_cpu=0 that_cpu=1 is_idle_task=1
    20.00%  (ffffffffa5e46bc9) this_cpu=3 that_cpu=0 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=0 that_cpu=2 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=0 that_cpu=3 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=1 that_cpu=0 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=2 that_cpu=3 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=3 that_cpu=1 is_idle_task=1
    10.00%  (ffffffffa5e46bc9) this_cpu=3 that_cpu=2 is_idle_task=1


#
# (Tip: Skip collecting build-id when recording: perf record -B)
#
bala@ubuntu-vm-1:~/source/linux/tools/perf$
```

Why `is_idle_task` is always 1 during and IPI?! More on later post ;).

# References
 * https://github.com/iovisor/bpftrace/issues/792
 * https://gcc.gnu.org/onlinedocs/gcc/Function-Specific-Option-Pragmas.html
 * https://news.ycombinator.com/item?id=4711571
 * https://www.cyberciti.biz/tips/compiling-linux-kernel-26.html
 * https://tldp.org/LDP/lame/LAME/linux-admin-made-easy/kernel-custom.html
 * https://stackoverflow.com/questions/28136815/linux-kernel-how-to-obtain-a-particular-version-right-upto-sublevel
 * https://git-scm.com/docs/git-fetch
 * https://www.quora.com/How-do-I-compile-a-Linux-perf-tool-with-all-features-For-Linux-4-0-on-Ubuntu
 * https://serverfault.com/questions/251134/how-to-compile-the-kernel-with-debug-symbols
