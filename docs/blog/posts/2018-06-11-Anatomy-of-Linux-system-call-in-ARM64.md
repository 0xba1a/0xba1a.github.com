---
title: Anatomy of Linux system call in ARM64
date: 2018-06-11 20:11:46
thumbnail: "/images/anatomy_of_system_call.png"
description: "Architecture of Linux system call. Explains ARM64 exception levels and privilege of each level. Walks through Linux code path related to interrupt vectors and instructions that handle the system call. Bit of assembly code understanding is necessary. Explains every single instruction between `printf` and `sys_write`."
tags:
 - system call
 - Linux
 - gdb
 - ARM
 - ARMv8
---

The purpose of an Operating System is to run user applications. But the OS cannot provide user applications full control due to security reasons. So to do some privileged operations applications should ask OS to do the job on behalf of themselves. The primary interaction mechanism in Linux and similar Operating Systems is System Call. In this article we are going to see the anatomy of Linux System Calls in ARM64 architecture. Most people working with applications doesn't require it. It is only for those who have a printed copy of `Hitchhiker's guide to the galaxy`. DON'T PANIC!

ARMv8 has four exception levels - EL0 to EL3. EL0 has lowest privilege where user applications run. EL3 has the highest privilege for Secure Monitor firmware (usually proprietary). Hypervisor runs in EL2 for virtualisation platforms. And our beloved Linux kernel runs in EL1. Elevation from one exception level to next exception level are achieved by setting exceptions. These exceptions will be set by one level and the next level will handle it. Explaining all types of exceptions is out of the scope of this article.

The instruction used to set a synchronous exception [used for system call mechanism] to elevate from EL0 to EL1 is `svc` - supervisor call. Thus an application runs in Linux should issue `svc` with registers set with appropriate values. To know what are those appropriate values, Lets see how kernel handles `svc`.

# Kernel Part
<div style="color:#3273dc">
<span style="font-weight: bold;">NOTE:</span> All the code references given in this post are from Linux-4.9.57.
</div>

### Vector table definition in Kernel
As I have mentioned already, there are multiple exceptions can be set by applications [EL0] which will be taken by Kernel [EL1]. The handlers for these exceptions are stored in a vector table. In ARMv8 the register that mentions the base address of that vector table is `VBAR_EL1` [Vector Base Address Register for EL1].

{% blockquote ARM infocenter %}
When an exception occurs, the processor must execute handler code which corresponds to the exception. The location in memory where the handler is stored is called the exception vector. In the ARM architecture, exception vectors are stored in a table, called the exception vector table. Each Exception level has its own vector table, that is, there is one for each of EL3, EL2 and EL1. The table contains instructions to be executed, rather than a set of addresses. Vectors for individual exceptions are located at fixed offsets from the beginning of the table. The virtual address of each table base is set by the Vector Based Address Registers VBAR_EL3, VBAR_EL2 and VBAR_EL1.
{% endblockquote %}

As explained above the exception-handlers reside in a continuous memory and each vector spans up to 32 instructions long. Based on type of the exception, the execution will start from an instruction in a particular offset from the base address `VBAR_EL1`. Below is the ARM64 vector table. For example when an synchronous exception is set from EL0 is set, the handler at `VBAR_EL1 +0x400` will execute to handle the exception.

Offset from VBAR_EL1          | Exception type        | Exception set level
------------------------------|-----------------------|---------------------------
+0x000                        | Synchronous           | Current EL with SP0
+0x080                        | IRQ/vIRQ              | "
+0x100                        | FIQ/vFIQ              | "
+0x180                        | SError/vSError        | "
+0x200                        | Synchronous           | Current EL with SPx
+0x280                        | IRQ/vIRQ              | "
+0x300                        | FIQ/vFIQ              | "
+0x380                        | SError/vSError        | "
+0x400                        | Synchronous           | Lower EL using ARM64
+0x480                        | IRQ/vIRQ              | "
+0x500                        | FIQ/vFIQ              | "
+0x580                        | SError/vSError        | "
+0x600                        | Synchronous           | Lower EL with ARM32
+0x680                        | IRQ/vIRQ              | "
+0x700                        | FIQ/vFIQ              | "
+0x780                        | SError/vSError        | "

Linux defines the vector table at `arch/arm64/kernel/entry.S +259`. Each `ventry` is 32 instructions long. As an instruction in ARMv8 is 4 bytes long, next `ventry` will start at +0x80 of current `ventry`.
```
ENTRY(vectors)
    ventry    el1_sync_invalid           // Synchronous EL1t
    ventry    el1_irq_invalid            // IRQ EL1t
    ventry    el1_fiq_invalid            // FIQ EL1t
    ventry    el1_error_invalid          // Error EL1t

    ventry    el1_sync                   // Synchronous EL1h
    ventry    el1_irq                    // IRQ EL1h
    ventry    el1_fiq_invalid            // FIQ EL1h
    ventry    el1_error_invalid          // Error EL1h

    ventry    el0_sync                   // Synchronous 64-bit EL0
    ventry    el0_irq                    // IRQ 64-bit EL0
    ventry    el0_fiq_invalid            // FIQ 64-bit EL0
    ventry    el0_error_invalid          // Error 64-bit EL0

#ifdef CONFIG_COMPAT
    ventry    el0_sync_compat            // Synchronous 32-bit EL0
    ventry    el0_irq_compat             // IRQ 32-bit EL0
    ventry    el0_fiq_invalid_compat     // FIQ 32-bit EL0
    ventry    el0_error_invalid_compat   // Error 32-bit EL0
#else
    ventry    el0_sync_invalid           // Synchronous 32-bit EL0
    ventry    el0_irq_invalid            // IRQ 32-bit EL0
    ventry    el0_fiq_invalid            // FIQ 32-bit EL0
    ventry    el0_error_invalid          // Error 32-bit EL0
#endif
END(vectors)
```
And loads the vector table into `VBAR_EL1` at `arch/arm64/kernel/head.S +433`.
```
    adr_l   x8, vectors                 // load VBAR_EL1 with virtual
    msr     vbar_el1, x8                // vector table address
    isb
```
`VBAR_EL1` is an system register. So it cannot be accessed directly. Special system instructions `msr` and `mrs` should be used manipulate system registers.

Instruction         | Description
--------------------|-------------------------------------------------------
`adr_l x8, vector`  | loads the address of vector table into general purpose register `X8`
`msr vbar_el1, x8`  | moves value in `X8` to system register `VBAR_EL1`
`isb`               | instruction sync barrier

### System call flow in Kernel

Now lets see what happens when an application issues the instruction `svc`. From the table, we can see for AArch64 synchronous exception from lower level, the offset is `+0x400`. In the Linux vector definition `VBAR_EL1+0x400` is `el0_sync`. Lets go to the `el0_sync` definition at `arch/arm64/kernel/entry.S +458`
```
el0_sync:
    kernel_entry 0
    mrs    x25, esr_el1                     // read the syndrome register
    lsr    x24, x25, #ESR_ELx_EC_SHIFT      // exception class
    cmp    x24, #ESR_ELx_EC_SVC64           // SVC in 64-bit state
    b.eq    el0_svc
    cmp    x24, #ESR_ELx_EC_DABT_LOW        // data abort in EL0
    b.eq    el0_da
    cmp    x24, #ESR_ELx_EC_IABT_LOW        // instruction abort in EL0
    b.eq    el0_ia
    cmp    x24, #ESR_ELx_EC_FP_ASIMD        // FP/ASIMD access
    b.eq    el0_fpsimd_acc
    cmp    x24, #ESR_ELx_EC_FP_EXC64        // FP/ASIMD exception
    b.eq    el0_fpsimd_exc
    cmp    x24, #ESR_ELx_EC_SYS64           // configurable trap
    b.eq    el0_sys
    cmp    x24, #ESR_ELx_EC_SP_ALIGN        // stack alignment exception
    b.eq    el0_sp_pc
    cmp    x24, #ESR_ELx_EC_PC_ALIGN        // pc alignment exception
    b.eq    el0_sp_pc
    cmp    x24, #ESR_ELx_EC_UNKNOWN         // unknown exception in EL0
    b.eq    el0_undef
    cmp    x24, #ESR_ELx_EC_BREAKPT_LOW     // debug exception in EL0
    b.ge    el0_dbg
    b    el0_inv
```
The subroutine is nothing but a bunch of `if` conditions. The synchronous exception can have multiple reasons which will be  stored in the syndrome register `esr_el1`. Compare the value in syndrome register with predefined macros and branch to the corresponding subroutine.

Instruction                         | Description
------------------------------------|--------------
`kernel_entry 0`                    | It is a macro defined at<br/>`arch/arm64/kernel/entry.S +71`. It stores<br/>all the general purpose registers into<br/>CPU stack as the sys_* function<br/>expects its arguments from<br/>CPU stack only
`mrs x25, esr_el1`                  | Move system register `esr_el1` to general<br/>purpose register `X25`. `esr_el1` is the exception <br/>syndrome register. It will have the syndrome code <br/>that caused the exception.
`lsr x24, x25, #ESR_ELx_EC_SHIFT`   | Left shift `X25` by `ESR_ELx_EC_SHIFT` bits <br/>and store the result in `X24`
`cmp x24, #ESR_ELx_EC_SVC64`        | Compare the value in `X24` with `ESR_ELx_EC_SVC64`. <br/>If both are equal `Z` bit will be set in `NZCV` <br/>special purpose register.
`b.eq el0_svc`                      | If `Z` flag is set in `NZCV`, branch to `el0_svc`. <br/>It is just `b` not `bl`. So the control will not come <br/>back to caller. The condition check will happen until <br/>it finds the appropriate reason. If all are wrong `el0_inv` <br/>will be called.
    
In a system call case, control will be branched to `el0_svc`. It is defined at `arm64/kernel/entry.S +742` as follows
```
/*
 * SVC handler.
 */
    .align    6
el0_svc:
    adrp    stbl, sys_call_table            // load syscall table pointer
    uxtw    scno, w8                        // syscall number in w8
    mov     sc_nr, #__NR_syscalls
el0_svc_naked:                              // compat entry point
    stp     x0, scno, [sp, #S_ORIG_X0]      // save the original x0 and syscall number    enable_dbg_and_irq
    ct_user_exit 1

    ldr     x16, [tsk, #TI_FLAGS]           // check for syscall hooks
    tst     x16, #_TIF_SYSCALL_WORK
    b.ne    __sys_trace
    cmp     scno, sc_nr                     // check upper syscall limit
    b.hs    ni_sys
    ldr     x16, [stbl, scno, lsl #3]       // address in the syscall table
    blr     x16                             // call sys_* routine
    b       ret_fast_syscall
ni_sys:
    mov     x0, sp
    bl      do_ni_syscall
    b       ret_fast_syscall
ENDPROC(el0_svc)
```
Before going through the code, let me introduce the aliases
`arch/arm64/kernel/entry.S +229`
```
/*
 * These are the registers used in the syscall handler, and allow us to
 * have in theory up to 7 arguments to a function - x0 to x6.
 *
 * x7 is reserved for the system call number in 32-bit mode.
 */
sc_nr   .req    x25        // number of system calls
scno    .req    x26        // syscall number
stbl    .req    x27        // syscall table pointer
tsk     .req    x28        // current thread_info
```
Now lets walk through function `el0_svc`,

Instruction                         | Description
------------------------------------|------------
`adrp stbl, sys_call_table`         | I'll come to the `sys_call_table` in next section. <br/>It is the table indexed with syscall number and <br/>corresponding function. It has to be in a <br/>4K aligned memory. Thus this instruction adds the top <br/>22-bits of `sys_call_table` address with top 52-bit of <br/>`PC` (program counter) and stores the value at `stbl`. <br/>Actually it forms the PC-relative address to 4KB page.
`uxtw scno, w8`                     | unsigned extract from 32-bit word. <br/>Read 32-bit form of General purpose register `X8` <br/>and store it in `scno`
`mov sc_nr, #__NR_syscalls`         | Load `sc_nr` with number of system calls
`stp x0, scno, [sp, #S_ORIG_X0]`    | Store a pair of registers `X0` and `scno` <br/>at the memory location stack-pointer + `S_ORIG_X0`. <br/>Value of `S_ORIG_X0` is not really important. <br/>As long as stack-pointer is not modified, <br/>we can access the stored values anytime.
`enable_dbg_and_irq`                | it is a macro defined at <br/>`arch/arm64/include/asm/assembler.h +88`. <br/>It actually enables IRQ and <br/>debugging by setting appropriate <br/>value at special purpose register `DAIF`.
`ct_user_exit 1`                    | another macro not much important <br/>unless you bother about `CONFIG_CONTEXT_TRACKING`
`ldr x16, [tsk, #TI_FLAGS]`<br/> `tst x16, #_TIF_SYSCALL_WORK` <br/>`b.ne __sys_trace` | These instructions are related to syscall hooks. <br/>If syscall hooks are set, call `__sys_trace`. <br/>For those who got confused about `b.ne` like me, <br/>`.ne` only check whether `Z` flag is non zero. <br/>`tst` instruction does an bitwise AND of both <br/>operands. If both are equal, `Z` flag will be non zero.
`cmp scno, sc_nr` <br/>`b.hs ni_sys`    | Just an error check. If the <br/>syscall number is greater than<br/>`sc_nr` go to `ni_sys`
`ldr x16, [stbl, scno, lsl #3]`     | Load the address of corresponding <br/>`sys_*` function into `X16`.<br/>Will explain detail in next section
`blr x16`                           | subroutine call to the actual `sys_*` <br/>function
`b ret_fast_syscall`                | Maybe a house-keeping function. <br/>Control will not flow further down.

### sys_call_table
It is nothing but an array of function pointer indexed with the system call number. It has to be placed in an 4K aligned memory. For ARM64 `sys_call_table` is defined at `arch/arm64/kernel/sys.c +55`.
```
#undef __SYSCALL
#define __SYSCALL(nr, sym)  [nr] = sym,

/*
 * The sys_call_table array must be 4K aligned to be accessible from
 * kernel/entry.S.
 */
void * const sys_call_table[__NR_syscalls] __aligned(4096) = {
    [0 ... __NR_syscalls - 1] = sys_ni_syscall,
#include <asm/unistd.h>
};
```
 * `__NR_syscalls` defines the number of system call. This varies from architecture to architecture.
 * Initially all the system call numbers were set `sys_ni_syscall` - not implemented system call. If a system call is removed, its system call number will not be reused. Instead it will be assigned with `sys_ni_syscall` function.
 * And the include goes like this `arch/arm64/include/asm/unistd.h` -> `arch/arm64/include/uapi/asm/unistd.h` -> `include/asm-generic/unistd.h` -> `include/uapi/asm-generic/unistd.h`. The last file has the definition of all system calls. For example the `write` system call is defined here as
```
#define __NR_write 64
__SYSCALL(__NR_write, sys_write)
```
 * The `sys_call_table` is an array of function pointers. As in ARM64 a function pointer is 8 bytes long, to calculate the address of actual system call, system call number `scno` is left shifted by 3 and added with system call table address `stbl` in the `el0_svc` subroutine - `ldr x16, [stbl, scno, lsl #3]`

### System call definition
Each system call is defined with a macro `SYSCALL_DEFINEn` macro. `n` is corresponding to the number of arguments the system call accepts. For example the `write` is implemented at `fs/read_write.c +599`
```
SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf,
        size_t, count)
{
    struct fd f = fdget_pos(fd);
    ssize_t ret = -EBADF;

    if (f.file) {
        loff_t pos = file_pos_read(f.file);
        ret = vfs_write(f.file, buf, count, &pos);
        if (ret >= 0)
            file_pos_write(f.file, pos);
        fdput_pos(f);
    }

    return ret;
}
```
This macro will expand into `sys_write` function definition and other aliases functions as mentioned in this [LWN article](https://lwn.net/Articles/604287/). The expanded function will have the compiler directive `asmlinkage` set. It instructs the compiler to look for arguments in CPU stack instead of registers. This is to implement system calls architecture independent. That's why `kernel_entry` macro in `el0_sync` pushed all general purpose registers into stack. In ARM64 case registers `X0` to `X7` will have the arguments.

# Application part

An user application should do following steps to make a system call
 * Set the lower 32-bit of general purpose register `X8` with appropriate system call number - `el0_svc` loads the system call number from `W0`
 * And then issue the `svc` instruction
 * Now kernel implemented `sys_*` function will run on behalf of the application

Normally all the heavy lifting will be done by the `glibc` library. Dependency on `glibc` also makes the application portable across platforms having `glibc` support. Each platform will have different system call number and `glibc` takes care of that while compiling.

### `printf` implementation

In this section we'll see how `glibc` implements the `printf` system call. Going deep into all `glibc` macro expansion is out of scope of this post. End of the day `printf` has to be a `write` towards the `stdout` file. Lets see that.

Wrote a simple hello world program.
```sh
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ cat syscall.c
#include <stdio.h>

int main()
{
    printf("Hello world!\n");
    return 0;
}
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ 
```
Compile it with symbols and export it to an ARM64 target [Raspberry Pi 3 in my case]. Run it in the target with `gdbserver` and connect remote GDB as mentioned in previous [post](http://eastrivervillage.com/debugging-application-with-cross-gdb-yocto/).

The last function in `glibc` that issues `svc` is `__GI___libc_write`. I got this with the help of `GDB`. If you really want to go through `glibc` code to trace this function all the way from `printf`, prepare yourself. Here is the `GDB` backtrace output.
```
(gdb) bt
#0  __GI___libc_write (fd=1, buf=0x412260, nbytes=13) at /usr/src/debug/glibc/2.26-r0/git/sysdeps/unix/sysv/linux/write.c:26
#1  0x0000007fb7eeddcc in _IO_new_file_write (f=0x7fb7fcd540 <_IO_2_1_stdout_>, data=0x412260, n=13) at /usr/src/debug/glibc/2.26-r0/git/libio/fileops.c:1255
#2  0x0000007fb7eed160 in new_do_write (fp=0x7fb7fcd540 <_IO_2_1_stdout_>, data=0x412260 "Hello world!\n", to_do=to_do@entry=13) at /usr/src/debug/glibc/2.26-r0/git/libio/fileops.c:510
#3  0x0000007fb7eeefc4 in _IO_new_do_write (fp=fp@entry=0x7fb7fcd540 <_IO_2_1_stdout_>, data=<optimized out>, to_do=13) at /usr/src/debug/glibc/2.26-r0/git/libio/fileops.c:486
#4  0x0000007fb7eef3f0 in _IO_new_file_overflow (f=0x7fb7fcd540 <_IO_2_1_stdout_>, ch=10) at /usr/src/debug/glibc/2.26-r0/git/libio/fileops.c:851
#5  0x0000007fb7ee3d78 in _IO_puts (str=0x400638 "Hello world!") at /usr/src/debug/glibc/2.26-r0/git/libio/ioputs.c:41
#6  0x0000000000400578 in main () at syscall.c:5
```
 Lets us see the assembly instructions of `__GI_libc_write` function in GDB TUI.
```
   |0x7fb7f407a8 <__GI___libc_write>        stp    x29, x30, [sp, #-48]!
  >│0x7fb7f407ac <__GI___libc_write+4>      adrp   x3, 0x7fb7fd1000 <__libc_pthread_functions+184>
   │0x7fb7f407b0 <__GI___libc_write+8>      mov    x29, sp
   │0x7fb7f407b4 <__GI___libc_write+12>     str    x19, [sp, #16]
   │0x7fb7f407b8 <__GI___libc_write+16>     sxtw   x19, w0
   │0x7fb7f407bc <__GI___libc_write+20>     ldr    w0, [x3, #264]
   │0x7fb7f407c0 <__GI___libc_write+24>     cbnz   w0, 0x7fb7f407f0 <__GI___libc_write+72>
   │0x7fb7f407c4 <__GI___libc_write+28>     mov    x0, x19
   │0x7fb7f407c8 <__GI___libc_write+32>     mov    x8, #0x40                       // #64
   │0x7fb7f407cc <__GI___libc_write+36>     svc    #0x0
   .
   .
   .
```

Instruction                         | Description
------------------------------------|------------
stp    x29, x30, [sp, #-48]!        | Increment stack pointer and back-up<br/>frame-pointer and link-register
adrp   x3, 0x7fb7fd1000             | Load the PC related address<br/>of `SINGLE_THREAD_P`
mov    x29, sp                      | Move current stack-pointer<br/>into frame-pointer
str    x19, [sp, #16]               | Backup `X19` in stack
sxtw   x19, w0                      | Backup `W0` into `X19`<br/> the first parameter<br/>This function has 3 parameters<br/>So they will be in `X0`, `X1`<br/>and`X2`. Thus they have to be<br/>backed-up before using
ldr    w0, [x3, #264]               | Load the global into `W0`<br/>This global tells whether<br/>it is a multi-threaded<br/>program
cbnz   w0, 0x7fb7f407f0             | Conditional branch if `W0`<br/>is non zero<br/>Actually a jump in case of<br/>multi-threaded program.<br/>Our case is single threaded.<br/>So fall through
mov    x0, x19                      | Restore `X19` into `X0`.<br/>The first argument.<br/>Here the `fd`
mov    x8, #0x40                    | Load `X8` with the value `0x40`.<br/>Kernel will look<br/>for the system call number<br/>at `X8`. So we load it with 64<br/>which is the system call<br/>number of `write`
svc    #0x0                         | All set. Now <br/>`supervisor call`

From this point the almighty Linux kernel takes control. Pretty long article ahh!. At last the improbable drive comes to equilibrium as you complete this.

# References
 * http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0024a/ch09s01s01.html
 * http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0024a/CHDEEDDC.html
 * https://lwn.net/Articles/604287/
 * https://courses.cs.washington.edu/courses/cse469/18wi/Materials/arm64.pdf
 * http://www.osteras.info/personal/2013/10/11/hello-world-analysis.html
 * And the Linux kernel source. Thanks to https://elixir.bootlin.com
