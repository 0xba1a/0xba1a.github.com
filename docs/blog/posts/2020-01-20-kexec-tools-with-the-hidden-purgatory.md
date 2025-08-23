---
title: kexec - A travel to the purgatory
date: 2020-01-20 17:35:13
thumbnail: "/images/purgatory.jpg"
description: "A story about how I fixed the boot delay in an ARM64 system. This article walks through Linux reboot / kexec code path, a bit of kexec-tools source and
importantly how kexec-tools injects its code in your boot path."
tags:
 - linux
 - ARMv8
 - kernel
 - crazy debugging
---

This is one of the unforgettable experience in my engineering life. Last year when we brought-up a ARM64 based platform, we faced lots of hurdles. But this one was very interesting and had lots of surprises. Though this triaging effort tolled multiple frustrating days, I had very good learning. I had to understand `kexec_load` system call, kernel reboot path and kernel early boot path to root cause the issue. We were using 4.14.19 kernel for the bring-up. But I'll give code samples from 5.4.14 as it is latest. There is no much code difference.

The boot flow of our new platform was `uboot --> service Linux --> main Linux`. We had `service Linux` to offload major hardware initialization work from the bootloader. Also to keep that code platform agnostic [same `service Linux` ran on x86 platforms too].

To make the jump from `service Linux` to `main Linux` we used `kexec`. The problem here was it took almost 2 minutes for the `main Linux` to start booting. This was a significant difference while comparing with x86 platforms. And this kind of delay would definitely annoy our customers.

After hundreds of rebuilds and thousands of print statements, I found the cause and fixed the issue. Now that 2 minutes delay was reduced to 4 seconds. Let me just tell you the code flow rather dump you with my debugging methods. I kept deferring from writing this article for almost an year. Because I was afraid of the code references and connecting them to show the flow.

# Code reference
| package     | version                                                        | repo                                            |
| ---         | ---                                                            | ---                                             |
| kexec-tools | today's latest master 2c9f26ed20a791a7df0182ba82e93abb52f5a615 | https://github.com/horms/kexec-tools            |
| linux       | 5.4.14                                                         | https://elixir.bootlin.com/linux/v5.4.14/source |


<div style="color:#3273dc">
<span style="font-weight: bold;">NOTE:</span> I copied code of entire functions for reference. But explained only necessary lines for better understanding.
</div>

# User story
Login to the `service Linux`. Load `main Linux`'s kernel and initrd. The current `device-tree` will be taken for `main Linux` too. Boot to `main Linux`.

```sh
# kexec -l /main/vmlinuz --initrd=/main/initrd.img
# kexec -e
```

As I mentioned earlier there was 2 mins delay between last `Bye` from `service Linux` and first message from `main Linux`. So I started looking for time consuming operations between those two prints.

# kexec-tools `kexec -e`

`kexec -e` calls `reboot()` system call with `LINUX_REBOOT_CMD_KEXEC` as argument.
``` C
kexec/kexec.c my_exec +900:
---
reboot(LINUX_REBOOT_CMD_KEXEC);
```
<br/>
# Kernel reboot path
### reboot system call
Reboot system call calls `kernel_kexec()` for the argument `LINUX_REBOOT_CMD_KEXEC`. You can refer to my earlier article [Anatomy of Linux system call in ARM64](http://eastrivervillage.com/Anatomy-of-Linux-system-call-in-ARM64/) to understand how arguments are passed from user space to kernel space.
``` C
kernel/reboot.c SYSCALL_DEFINE4(reboot, ...) +380:
---
#ifdef CONFIG_KEXEC_CORE
	case LINUX_REBOOT_CMD_KEXEC:
		ret = kernel_kexec();
		break;
#endif
```

### kernel_kexec()
`kernel_kexec()` at `kernel/kexec_core.c +1119` does following things.
```C
kernel/kexec_core.c +1119
---
/*
 * Move into place and start executing a preloaded standalone
 * executable.  If nothing was preloaded return an error.
 */
int kernel_kexec(void)
{
	int error = 0;

	if (!mutex_trylock(&kexec_mutex))
		return -EBUSY;
	if (!kexec_image) {
		error = -EINVAL;
		goto Unlock;
	}

#ifdef CONFIG_KEXEC_JUMP
	if (kexec_image->preserve_context) {
		lock_system_sleep();
		pm_prepare_console();
		error = freeze_processes();
		if (error) {
			error = -EBUSY;
			goto Restore_console;
		}
		suspend_console();
		error = dpm_suspend_start(PMSG_FREEZE);
		if (error)
			goto Resume_console;
		/* At this point, dpm_suspend_start() has been called,
		 * but *not* dpm_suspend_end(). We *must* call
		 * dpm_suspend_end() now.  Otherwise, drivers for
		 * some devices (e.g. interrupt controllers) become
		 * desynchronized with the actual state of the
		 * hardware at resume time, and evil weirdness ensues.
		 */
		error = dpm_suspend_end(PMSG_FREEZE);
		if (error)
			goto Resume_devices;
		error = suspend_disable_secondary_cpus();
		if (error)
			goto Enable_cpus;
		local_irq_disable();
		error = syscore_suspend();
		if (error)
			goto Enable_irqs;
	} else
#endif
	{
		kexec_in_progress = true;
		kernel_restart_prepare(NULL);
		migrate_to_reboot_cpu();

		/*
		 * migrate_to_reboot_cpu() disables CPU hotplug assuming that
		 * no further code needs to use CPU hotplug (which is true in
		 * the reboot case). However, the kexec path depends on using
		 * CPU hotplug again; so re-enable it here.
		 */
		cpu_hotplug_enable();
		pr_emerg("Starting new kernel\n");
		machine_shutdown();
	}

	machine_kexec(kexec_image);

#ifdef CONFIG_KEXEC_JUMP
	if (kexec_image->preserve_context) {
		syscore_resume();
 Enable_irqs:
		local_irq_enable();
 Enable_cpus:
		suspend_enable_secondary_cpus();
		dpm_resume_start(PMSG_RESTORE);
 Resume_devices:
		dpm_resume_end(PMSG_RESTORE);
 Resume_console:
		resume_console();
		thaw_processes();
 Restore_console:
		pm_restore_console();
		unlock_system_sleep();
	}
#endif

 Unlock:
	mutex_unlock(&kexec_mutex);
	return error;
}
```

| line   | code                                   | explanation                                                                                                                                             |
| ------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1123   | `mutex_trylock(&kexec_mutex)`          | This lock is held to avoid multiple entrance into kexec                                                                                                 |
| 1131   | `if (kexec_image->preserve_context) {` | Something related to keep the device status not disturbed during kexec. We were not using it. Moving to else part.                                      |
| 1165   | `migrate_to_reboot_cpu()`              | Continue execution from logical CPU 0. Only one control path is valid during reboot and startup. That will be executed from CPU-0.                      |
| 1175   | `machine_shutdown()`                   | Don't get confused by the name of this function. Its just a wrapper around `disable_nonboot_cpus()` in arm64. It does nothing but disables other CPUs   |
| 1178   | `machine_kexec(kexec_image)`           | Execution continues passing `kexec_image` as argument. This call should not return.                                                                     |

### machine_kexec()
```C
arch/arm64/kernel/machine_kexec.c +144
---
/**
 * machine_kexec - Do the kexec reboot.
 *
 * Called from the core kexec code for a sys_reboot with LINUX_REBOOT_CMD_KEXEC.
 */
void machine_kexec(struct kimage *kimage)
{
	phys_addr_t reboot_code_buffer_phys;
	void *reboot_code_buffer;
	bool in_kexec_crash = (kimage == kexec_crash_image);
	bool stuck_cpus = cpus_are_stuck_in_kernel();

	/*
	 * New cpus may have become stuck_in_kernel after we loaded the image.
	 */
	BUG_ON(!in_kexec_crash && (stuck_cpus || (num_online_cpus() > 1)));
	WARN(in_kexec_crash && (stuck_cpus || smp_crash_stop_failed()),
		"Some CPUs may be stale, kdump will be unreliable.\n");

	reboot_code_buffer_phys = page_to_phys(kimage->control_code_page);
	reboot_code_buffer = phys_to_virt(reboot_code_buffer_phys);

	kexec_image_info(kimage);

	pr_debug("%s:%d: control_code_page:        %p\n", __func__, __LINE__,
		kimage->control_code_page);
	pr_debug("%s:%d: reboot_code_buffer_phys:  %pa\n", __func__, __LINE__,
		&reboot_code_buffer_phys);
	pr_debug("%s:%d: reboot_code_buffer:       %p\n", __func__, __LINE__,
		reboot_code_buffer);
	pr_debug("%s:%d: relocate_new_kernel:      %p\n", __func__, __LINE__,
		arm64_relocate_new_kernel);
	pr_debug("%s:%d: relocate_new_kernel_size: 0x%lx(%lu) bytes\n",
		__func__, __LINE__, arm64_relocate_new_kernel_size,
		arm64_relocate_new_kernel_size);

	/*
	 * Copy arm64_relocate_new_kernel to the reboot_code_buffer for use
	 * after the kernel is shut down.
	 */
	memcpy(reboot_code_buffer, arm64_relocate_new_kernel,
		arm64_relocate_new_kernel_size);

	/* Flush the reboot_code_buffer in preparation for its execution. */
	__flush_dcache_area(reboot_code_buffer, arm64_relocate_new_kernel_size);

	/*
	 * Although we've killed off the secondary CPUs, we don't update
	 * the online mask if we're handling a crash kernel and consequently
	 * need to avoid flush_icache_range(), which will attempt to IPI
	 * the offline CPUs. Therefore, we must use the __* variant here.
	 */
	__flush_icache_range((uintptr_t)reboot_code_buffer,
			     arm64_relocate_new_kernel_size);

	/* Flush the kimage list and its buffers. */
	kexec_list_flush(kimage);

	/* Flush the new image if already in place. */
	if ((kimage != kexec_crash_image) && (kimage->head & IND_DONE))
		kexec_segment_flush(kimage);

	pr_info("Bye!\n");

	local_daif_mask();

	/*
	 * cpu_soft_restart will shutdown the MMU, disable data caches, then
	 * transfer control to the reboot_code_buffer which contains a copy of
	 * the arm64_relocate_new_kernel routine.  arm64_relocate_new_kernel
	 * uses physical addressing to relocate the new image to its final
	 * position and transfers control to the image entry point when the
	 * relocation is complete.
	 * In kexec case, kimage->start points to purgatory assuming that
	 * kernel entry and dtb address are embedded in purgatory by
	 * userspace (kexec-tools).
	 * In kexec_file case, the kernel starts directly without purgatory.
	 */
	cpu_soft_restart(reboot_code_buffer_phys, kimage->head, kimage->start,
#ifdef CONFIG_KEXEC_FILE
						kimage->arch.dtb_mem);
#else
						0);
#endif

	BUG(); /* Should never get here. */
}
```

| line      | code                                                                                                                                                                                                                                                                                     | explanation                                                                                                                                                                                                                                                                                                                                                              |
| ---       | ---                                                                                                                                                                                                                                                                                               | ---                                                                                                                                                                                                                                                                                                                                                                 |
| 148       | `bool in_kexec_crash = (kimage == kexec_crash_image)`                                                                                                                                                                                                                                             | Not true in our case. `kimage` is `kexec_image`                                                                                                                                                                                                                                                                                                                     |
| 149       | `bool stuck_cpus = cpus_are_stuck_in_kernel();`                                                                                                                                                                                                                                                   | Get number of stuck CPUs. Ideally it should be 0                                                                                                                                                                                                                                                                                                                    |
| 154       | <code>BUG_ON(!in_kexec_crash && (stuck_cpus &#124;&#124; (num_online_cpus() > 1)))</code>                                                                                                                                                                                                         | In non-crash situations, no CPU should be stuck and no CPU other than reboot CPU should be online.                                                                                                                                                                                                                                                            |
| 158       | `reboot_code_buffer_phys = page_to_phys(kimage->control_code_page)`                                                                                                                                                                                                                               | Get the physical page address from `kimage->control_code_page`. `arm64_relocate_new_kernel` function will be copied to this special location.                                                                                                                                                                                                                                                                                          |
| 159       | `reboot_code_buffer = phys_to_virt(reboot_code_buffer_phys)`                                                                                                                                                                                                                                      | Store the virtual address of same to continue working on that area.                                                                                                                                                                                                                                                                                                 |
| 179       | `memcpy(reboot_code_buffer, arm64_relocate_new_kernel, arm64_relocate_new_kernel_size)`                                                                                                                                                                                                           | Copies `arm64_relocate_new_kernel_size` routines address to the jump location. It is implemented in assembly. This is the routine that copies new kernel to its correct place. To make sure the memory where this routine resides doesn't get overwritten during the copy, it is copied inside `kexec_image` and executed from there. Never expected right? me too. |
| 183 - 199 | `__flush_dcache_area(reboot_code_buffer, arm64_relocate_new_kernel_size);`<br/>`__flush_icache_range((uintptr_t)reboot_code_buffer, arm64_relocate_new_kernel_size);`<br/>`kexec_list_flush(kimage);if ((kimage != kexec_crash_image) && (kimage->head & IND_DONE)) kexec_segment_flush(kimage);` | Flush necessary memory areas                                                                                                                                                                                                                                                                                                                                             |
| 201       | `pr_info("Bye!\n")`                                                                                                                                                                                                                                                                              | <b>The last print message from current kernel</b>                                                                                                                                                                                                                                                                                                                      |
| 203       | `local_daif_mask()`                                                                                                                                                                                                                                                                              | Disable all exceptions including interrupts. As we are entering into reboot path, don't expect any normal operations.                                                                                                                                                                                                                                            |
| 217       | `cpu_soft_restart(reboot_code_buffer_phys, kimage->head, kimage->start,0)`                                                                                                                                                                                                                        | This call won't return. `CONFIG_KEXEC_FILE` is not necessary in our case. The comment block above this call explains about purgatory code. But unfortunately that was not written when I was actually debugging this issue.                                                                                                                                  |


### cpu_soft_restart()
Its just a wrapper around `__cpu_soft_restart` which is an assembly routine. `el2_switch` would be set to 0 in our case.
```C
arch/arm64/kernel/cpu-reset.h +16
---
void __cpu_soft_restart(unsigned long el2_switch, unsigned long entry,
	unsigned long arg0, unsigned long arg1, unsigned long arg2);

static inline void __noreturn cpu_soft_restart(unsigned long entry,
					       unsigned long arg0,
					       unsigned long arg1,
					       unsigned long arg2)
{
	typeof(__cpu_soft_restart) *restart;

	unsigned long el2_switch = !is_kernel_in_hyp_mode() &&
		is_hyp_mode_available();
	restart = (void *)__pa_symbol(__cpu_soft_restart);

	cpu_install_idmap();
	restart(el2_switch, entry, arg0, arg1, arg2);
	unreachable();
}
```

An important call to note here is `cpu_install_idmap()`. This comes handy when MMU comes in or goes out. Linux kernel has a text section called `idmap.text`. `cpu_install_idmap()` will copy this section to two memory areas. There will be virtual memory mapping for one of these areas. The second area will be literal physical memory location of the virtual address. For example code in this section will be loaded at physical address 0x0 and 0x80000000. And the virtual address corresponding to 0x80000000 will be 0x0. It ensures continuous execution after MMU goes off. I'll write a separate article explaining in detail about this.

### __cpu_soft_restart()
```ASM
arch/arm64/kernel/cpu-reset.S +15
---
.text
.pushsection    .idmap.text, "awx"

/*
 * __cpu_soft_restart(el2_switch, entry, arg0, arg1, arg2) - Helper for
 * cpu_soft_restart.
 *
 * @el2_switch: Flag to indicate a switch to EL2 is needed.
 * @entry: Location to jump to for soft reset.
 * arg0: First argument passed to @entry. (relocation list)
 * arg1: Second argument passed to @entry.(physical kernel entry)
 * arg2: Third argument passed to @entry. (physical dtb address)
 *
 * Put the CPU into the same state as it would be if it had been reset, and
 * branch to what would be the reset vector. It must be executed with the
 * flat identity mapping.
 */
ENTRY(__cpu_soft_restart)
	/* Clear sctlr_el1 flags. */
	mrs	x12, sctlr_el1
	ldr	x13, =SCTLR_ELx_FLAGS
	bic	x12, x12, x13
	pre_disable_mmu_workaround
	msr	sctlr_el1, x12
	isb

	cbz	x0, 1f				// el2_switch?
	mov	x0, #HVC_SOFT_RESTART
	hvc	#0				// no return

1:	mov	x18, x1				// entry
	mov	x0, x2				// arg0
	mov	x1, x3				// arg1
	mov	x2, x4				// arg2
	br	x18
ENDPROC(__cpu_soft_restart)

.popsection
```

| line  | instruction                                                                                                                                            | explanation                                                                                                                        |
| ---   | ---                                                                                                                                                    | ---                                                                                                                                |
| 16    | `.pushsection    .idmap.text, "awx"`                                                                                                                   | As I mentioned earlier, this routine has to be pushed into `idmap.text` section. Because it is going to disable MMU.               |
| 34-38 | `mrs     x12, sctlr_el1`<br/>`ldr     x13, =SCTLR_ELx_FLAGS`<br/>`bic     x12, x12, x13`<br/>`pre_disable_mmu_workaround`<br/>`msr     sctlr_el1, x12` | Disable I,D cache and MMU                                                                                                          |
| 45-49 | `mov     x18, x1`<br/>`mov     x0, x2`<br/>`mov     x1, x3`<br/>`mov     x2, x4`<br/>`br      x18`                                                     | Set arguments and jump to `arm64_relocate_new_kernel routine`. In arm64 registers `x0`-`x6` are corresponding to first 7 arguments |

### arm64_relocate_new_kernel()
The assembly routine can be found at `arch/arm64/kernel/relocate_kernel.S +29`. It does nothing significant. It sets dtb address at `x0` and jumps to `kexec_image->entry` which is [supposed to be] pointing to the new kernel.

<br/>
# kexec load kernel path
As I expected if `kexec_image->entry` pointed to new kernel, I shouldn't had seen that delay. So to verify that I went through `kexec_load()` system call's path.

### kexec_load()

```C
kernel/kexec.c +232
---
SYSCALL_DEFINE4(kexec_load, unsigned long, entry, unsigned long, nr_segments,
		struct kexec_segment __user *, segments, unsigned long, flags)
{
	int result;

	result = kexec_load_check(nr_segments, flags);
	if (result)
		return result;

	/* Verify we are on the appropriate architecture */
	if (((flags & KEXEC_ARCH_MASK) != KEXEC_ARCH) &&
		((flags & KEXEC_ARCH_MASK) != KEXEC_ARCH_DEFAULT))
		return -EINVAL;

	/* Because we write directly to the reserved memory
	 * region when loading crash kernels we need a mutex here to
	 * prevent multiple crash  kernels from attempting to load
	 * simultaneously, and to prevent a crash kernel from loading
	 * over the top of a in use crash kernel.
	 *
	 * KISS: always take the mutex.
	 */
	if (!mutex_trylock(&kexec_mutex))
		return -EBUSY;

	result = do_kexec_load(entry, nr_segments, segments, flags);

	mutex_unlock(&kexec_mutex);

	return result;
}
```
It first checks permission with `kexec_load_check` function. Then it takes the `kexec_mutex` and goes into `do_kexec_load()`

### do_kexec_load()
``` C
kernel/kexec.c +106
---
static int do_kexec_load(unsigned long entry, unsigned long nr_segments,
		struct kexec_segment __user *segments, unsigned long flags)
{
	struct kimage **dest_image, *image;
	unsigned long i;
	int ret;

	if (flags & KEXEC_ON_CRASH) {
		dest_image = &kexec_crash_image;
		if (kexec_crash_image)
			arch_kexec_unprotect_crashkres();
	} else {
		dest_image = &kexec_image;
	}

	if (nr_segments == 0) {
		/* Uninstall image */
		kimage_free(xchg(dest_image, NULL));
		return 0;
	}
	if (flags & KEXEC_ON_CRASH) {
		/*
		 * Loading another kernel to switch to if this one
		 * crashes.  Free any current crash dump kernel before
		 * we corrupt it.
		 */
		kimage_free(xchg(&kexec_crash_image, NULL));
	}

	ret = kimage_alloc_init(&image, entry, nr_segments, segments, flags);
	if (ret)
		return ret;

	if (flags & KEXEC_PRESERVE_CONTEXT)
		image->preserve_context = 1;

	ret = machine_kexec_prepare(image);
	if (ret)
		goto out;

	/*
	 * Some architecture(like S390) may touch the crash memory before
	 * machine_kexec_prepare(), we must copy vmcoreinfo data after it.
	 */
	ret = kimage_crash_copy_vmcoreinfo(image);
	if (ret)
		goto out;

	for (i = 0; i < nr_segments; i++) {
		ret = kimage_load_segment(image, &image->segment[i]);
		if (ret)
			goto out;
	}

	kimage_terminate(image);

	/* Install the new kernel and uninstall the old */
	image = xchg(dest_image, image);

out:
	if ((flags & KEXEC_ON_CRASH) && kexec_crash_image)
		arch_kexec_protect_crashkres();

	kimage_free(image);
	return ret;
}
```

| line    | code                                                                    | explanation                                                                                                   |
| ---     | ---                                                                     | ---                                                                                                           |
| 113-120 |10-17                                                                    | Its a simple if block choosing image based on context                                                         |
| 135     | `ret = kimage_alloc_init(&image, entry, nr_segments, segments, flags);` | Create a new image segment. <b>The second argument is important. This jump address.</b> |
| 142     | `ret = machine_kexec_prepare(image);`                                   | Check whether any other CPUs are stuck                                                                        |
| 154-158 |52-57                                                                    | Copies segments passed from user space to kernel space.                                                       |
| 163     | `image = xchg(dest_image, image);`                                      | Replaces the older image with new one                                                                         |

<br/>
The `entry` argument passed to `kexec_alloc_init()` function is assigned to `kexec_image->start`.
```C
kernel/kexec.c kexec_alloc_init +60
---
image->start = entry;
```
This `kexec_image->start` is passed to `cpu_soft_restart()` as the third argument. If you follow the argument flow in kernel reboot path [as explained above], `arm64_relocate_new_kernel()` jumps to this address. So this must be the address of new kernel.
<br/>

# kexec-tools
Now I went into `kexec-tools` source. Call to `kexec_load()` is as below.
```C
 kexec/kexec.c my_load +821
---
	result = kexec_load(info.entry,
			    info.nr_segments, info.segment,
			    info.kexec_flags);
```
`info.entry` is passed as the first argument. As we've seen in previous section, this the jump address. Lets see what is there in `info.entry`. The code flow goes like this

```C
kexec/kexec.c main +1551
---
	result = my_load(type, fileind, argc, argv,
				kexec_flags, skip_checks, entry);
```
```C
kexec/kexec.c my_load +772
---
	result = file_type[i].load(argc, argv, kernel_buf, kernel_size, &info);
```
```C
kexec/arch/arm/kexec-arm.c +82
---
struct file_type file_type[] = {
	/* uImage is probed before zImage because the latter also accepts
	   uncompressed images. */
	{"uImage", uImage_arm_probe, uImage_arm_load, zImage_arm_usage},
	{"zImage", zImage_arm_probe, zImage_arm_load, zImage_arm_usage},
};

```
```C
kexec/arch/arm64/kexec-zImage-arm64.c zImage_arm64_load +212
---
	result = arm64_load_other_segments(info, kernel_segment
		+ arm64_mem.text_offset);

```
```C
kexec/arch/arm64/kexec-arm64.c arm64_load_other_segments +743
---
	info->entry = (void *)elf_rel_get_addr(&info->rhdr, "purgatory_start");
```

Address of a symbol called `purgatory_start` is assigned to `info->entry`. So <b>old kernel makes a jump to this `purgatory_start` not to the new kernel.</b> This was the big surprise. The last thing I expected was `kexec-tools` inserts a piece of code between kernels. I felt that I came close to the criminal.

# The purgatory

``` ASM
purgatory/arch/arm64/entry.S +9
---
.text

.globl purgatory_start
purgatory_start:

	adr	x19, .Lstack
	mov	sp, x19

	bl	purgatory

	/* Start new image. */
	ldr	x17, arm64_kernel_entry
	ldr	x0, arm64_dtb_addr
	mov	x1, xzr
	mov	x2, xzr
	mov	x3, xzr
	br	x17

size purgatory_start
```
`purgatory_start` is an assembly subroutine. It calls a function `purgatory` first. And then stores `arm64_dtb_addr` in register `x0` and jumps to `arm64_kernel_entry`. As arm64 kernel expects dtb address to be set in register `x0`, this jump is likely to be the jump to new kernel. We'll see where these symbols are set.

```C
kexec/arch/arm64/kexec-arm64.c arm64_load_other_segments +
---
	elf_rel_build_load(info, &info->rhdr, purgatory, purgatory_size,
		hole_min, hole_max, 1, 0);

	info->entry = (void *)elf_rel_get_addr(&info->rhdr, "purgatory_start");

	elf_rel_set_symbol(&info->rhdr, "arm64_kernel_entry", &image_base,
		sizeof(image_base));

	elf_rel_set_symbol(&info->rhdr, "arm64_dtb_addr", &dtb_base,
		sizeof(dtb_base));
```
As you see in the code snippet, `image_base` is `arm64_kernel_entry` and `dtb_base` is `arm64_dtb_addr`. If you go little above in the function, you'll see `image_base` and `dtb_base` are kernel address and device-tree address respectively. And the function `purgatory` is also loaded into the segments. The last function to check between two kernels is `purgatory`.

```C
purgatory/purgatory.c
---
void purgatory(void)
{
	printf("I'm in purgatory\n");
	setup_arch();
	if (!skip_checks && verify_sha256_digest()) {
		for(;;) {
			/* loop forever */
		}
	}
	post_verification_setup_arch();
}
```
`setup_arch()` and `post_verification_setup_arch()` are no-op for arm64. The actual delay is caused by `verify_sha256_digest()`. There is nothing wrong with this function. But it is being executed with I and D caches off. The option `skip_checks` was not introduced by the time I was debugging this issue. So we solved it by enabling I & D cache in `setup_arch()` and disabling it back in `post_verification_setup_arch()`.

At last I felt like my purgatory sentence was over.
