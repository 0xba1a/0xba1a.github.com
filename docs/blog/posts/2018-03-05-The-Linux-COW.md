---
title: The Linux COW
date: 2018-03-05 21:59:30
thumbnail: "/images/linux_cow.jpeg"
authors:
 - Bala
tags:
 - memory_management
 - fork
 - clone
 - getpid
 - cow
 - stack
 - virtual_memory
---

In our [last post](2018-02-27-Virtual-memory-to-Physical-memory.md) we have seen how to get the physical memory address using a virtual memory address. In this post lets try to check Copy on Write(*COW* from here) implemented by Linux. For those who try to recollect what is COW, on `fork()`, Linux will not immediately replicate the address space of parent process. As the child process is most-likely to load a new binary, copying the current process's address space will be of no use in many cases. So unless a write occurs (either by the parent process or by the child process), the physical address space mapped to both processes' virtual address space will be same. Please go through [Wikipedia](https://en.wikipedia.org/wiki/Copy-on-write) if you still have no clue.

So lets write a C program that forks itself and check whether both are sharing the physical address space. We are going to reuse the `get_paddr.c` function we wrote in our [previous post](2018-02-27-Virtual-memory-to-Physical-memory.md) to get the physical memory address.
``` C
#include <stdio.h>
#include <unistd.h> /* fork() */

int main()
{
	int a;

	fork();

	printf("my pid is %d, and the address to work is 0x%lx\n", getpid(), (unsigned long)&a);
	scanf("%d\n", &a);

	return 0;
}
```

Execute this program in one console and while it is waiting for the user input, go to the next console and get the physical address of variable `a` using our `get_paddr` binary.
``` sh
$ gcc specimen.c -o specimen
$ ./specimen
my pid is 5912, and the address to work is 0x7ffd055923e4
my pid is 5913, and the address to work is 0x7ffd055923e4

```
``` sh
$ sudo ./get_paddr 5912 0x7ffd055923e4
getting page number of virtual address 140724693181412 of process 5912
opening pagemap /proc/5912/pagemap
moving to 274852916368
physical frame address is 0x509c9
physical address is 0x509c93e4
$
$ sudo ./get_paddr 5913 0x7ffd055923e4
getting page number of virtual address 140724693181412 of process 5913
opening pagemap /proc/5913/pagemap
moving to 274852916368
physical frame address is 0x64d2a
physical address is 0x64d2a3e4
$
```

**OOPS!** The physical address is not same! How? Why? What's wrong? As per my beloved *Robert Love's Linux System Programming* book, the physical memory copy will occur only when a write occurs.

{% blockquote Robert Love , Linux System Programming %}
The MMU can intercept the write operation and raise an exception; the kernel, in response, will transparently create a new copy of the page for the writing process, and allow the write to continue against the new page. We call this approach *copy-on-write(COW). Effectively, processes are allowed read access to shared data, which saves space. But when a process wants to write to a shared page, it receives a unique copy of that page on fly, thereby allowing the kernel to act as if the process always had its own private copy. As copy-on write occurs on a page-by-page basis, with the technique a huge file may be efficiently shared among many processes, and the individual processes will receive unique physical pages only for those pages to which thy themselves write.
{% endblockquote %}

It is so clear that in our program `specimen.c` we never write to the only variable `a` unless the `scanf` completes. But we tested the *COW* before input-ing to the `scanf`. So by the time there would be no write and as per the design of *COW* physical memory space should be shared across both parent and child. Even though we have written the program carefully, sometimes the libraries, compiler and even the OS will act intelligently (stupid! They themselves says it) and fills us with surprise. So lets run the handy `strace` on our `specimen` binary to see what it actually does.
``` sh
$ strace ./specimen
execve("./specimen", ["./specimen"], [/* 47 vars */]) = 0
brk(NULL)                               = 0x55de17b57000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=94696, ...}) = 0
mmap(NULL, 94696, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7fdbad5b8000
close(3)                                = 0
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\340\22\2\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=1960656, ...}) = 0
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7fdbad5b6000
mmap(NULL, 4061792, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7fdbacfc9000
mprotect(0x7fdbad19f000, 2097152, PROT_NONE) = 0
mmap(0x7fdbad39f000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE,
		3, 0x1d6000) = 0x7fdbad39f000
mmap(0x7fdbad3a5000, 14944, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS,
		-1, 0) = 0x7fdbad3a5000
close(3)                                = 0
arch_prctl(ARCH_SET_FS, 0x7fdbad5b7500) = 0
mprotect(0x7fdbad39f000, 16384, PROT_READ) = 0
mprotect(0x55de15e94000, 4096, PROT_READ) = 0
mprotect(0x7fdbad5d0000, 4096, PROT_READ) = 0
munmap(0x7fdbad5b8000, 94696)           = 0
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD,
		child_tidptr=0x7fdbad5b77d0) = 5932
getpid()                                = 5931
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 2), ...}) = 0
my pid is 5932, and the address to work is 0x7fff9823b814
brk(NULL)                               = 0x55de17b57000
brk(0x55de17b78000)                     = 0x55de17b78000
write(1, "my pid is 5931, and the address "..., 58my pid is 5931, and the address
		to work is 0x7fff9823b814
) = 58
fstat(0, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 2), ...}) = 0
read(0,

```

As the program is waiting on `scanf`, it is waiting for user input in `read` system-call from `fd 0`, `stdin`. But the surprise part here is the `clone` system call. If you don't remember, read the `specimen.c` program once again. We never used the `clone` system call. And where is the `fork`? It seems `glibc` does something in a so called *intelligent* way. Lets read the `man pages` once again for pointers.
``` sh
C library/kernel differences
       Since  version  2.3.3, rather than invoking the kernel's fork() system call,
	   the glibc fork() wrapper that is provided as part of the NPTL threading
	   implementation invokes clone(2) with flags that provide the same effect as
	   the traditional system call.  (A call to fork() is equivalent to a call to
	   clone(2) specifying flags as just  SIGCHLD.) The glibc wrapper invokes any
	   fork handlers that have been established using pthread_atfork(3).
```

Doesn't look completely true. Though the `man page` says it calls `clone()` with flags as just `SIGCHLD`, `strace` uncovers the dirty truth - `flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD`. Nobody escapes from an armed system programmer.  Anyway this explains `clone` but no-`fork` surprise. But where comes the write that causes `COW`. See the last argument of `clone()` in `strace`, `child_tidptr=0x7fdbad5b77d0` some place holder stack variable passed to be filled. And what about the two flags other than `SIGCHLD`? Lets go to the `man pages` once again, but for `clone()` this time.
``` sh
    CLONE_CHILD_CLEARTID (since Linux 2.5.49)
        Clear (zero) the child thread ID at the location ctid in child memory when
		the child exits, and do a wakeup on the futex at that address.  The address
		involved  may be changed by the set_tid_address(2) system call.  This is
		used by threading libraries.

    CLONE_CHILD_SETTID (since Linux 2.5.49)
        Store the child thread ID at the location ctid in the child's memory.
		The store operation completes before clone() returns control to user space.
```

Ahh! The culprit is `CLONE_CHILD_SETTID`. It makes the OS to write thread ID of the child into its memory region. This write triggers the `COW` and gets the child a fresh copy of physical memory. Sorry Linux, I doubted you.

Okay. Lets modify our `specimen` with `clone()` and no `CLONE_CHILD_SETTID` flag.
``` C
/* clone() */
#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>

/* getpid() */
#include <sys/types.h>
#include <unistd.h>

#include <signal.h> /* SIGCHLD */
#include <stdlib.h> /* malloc() */

int run(void *arg)
{
	printf("my pid is %d and address to check is 0x%lx\n", getpid(), (unsigned long) arg);
	scanf("%d\n", (int *)arg);

	return 0;
}

int main()
{
	int a;
	void *stack_ptr = malloc(1024*1024);
	if (stack_ptr == NULL) {
		printf("No virtual memory available\n");
		return -1;
	}
	/*fork();*/

	clone(run, (stack_ptr + 1024*1024), CLONE_CHILD_CLEARTID|SIGCHLD, &a);
	if (a < 0)
		perror("clone failed. ");

	run(&a);

	return 0;
}
```

Unlike `fork()`, `clone()` doesn't start the execution of child from the point where it returns.Instead it takes a function as an argument and runs the function in child process. Here we are passing `run()` function which prints the `pid` and `virtual address` of the variable `a`. After `clone()` the parent also calls the function `run()` so that we'll get the `pid` of parent process.

Have you noted the second argument of `clone()`? It is the `stack` where child is going to execute. Again unlike `fork()`, `clone()` allows parent and child to share their resources like memory, signal handlers, open file descriptors, etc. As both cannot run in same `stack`, the parent process must allocate some space and give it to the child to use it as stack. `stack` grows downwards on all processors that run Linux, so the child-stack should point out to the topmost address of the memory region. That's why we are passing the end of `malloc`ed memory. Lets execute the program and see whether both process share the same physical memory space.
``` sh
$ ./specimen
my pid is 3129 and address to check is 0x7fff396fdd6c
my pid is 3130 and address to check is 0x7fff396fdd6c

```
``` sh
$ sudo ./get_paddr 3129 0x7fff396fdd6c
getting page number of virtual address 140734157020524 of process 3129
opening pagemap /proc/3129/pagemap
moving to 274871400424
physical frame address is 0x5e35a
physical address is 0x5e35ad6c
$
$ sudo ./get_paddr 3130 0x7fff396fdd6c
getting page number of virtual address 140734157020524 of process 3130
opening pagemap /proc/3130/pagemap
moving to 274871400424
physical frame address is 0x6a7cd
physical address is 0x6a7cdd6c
$
```

Different again! Linux can't be wrong. We should make sure we didn't mess-up anything. Are we sure there will be no write in stack after `clone()` call?
 * The child's execution starts at function `run()`
 * No variables are written until `scanf()` returns. Okay we'll complete our examination before providing any input.
 * But the functions? OOPS! After clone, parent calls `run()` and then `printf()` but child calls `printf()` directly. All these function calls will make write in stack which triggers *COW*.

Lets come-up with a different C-program with no function calls after `clone()` in both parent and child.

``` C
/* clone() */
#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>

/* getpid() */
#include <sys/types.h>
#include <unistd.h>

#include <signal.h> /* SIGCHLD */
#include <stdlib.h> /* malloc() */

#define STACK_LENGTH (1024*1024)

int run(void *arg)
{
	while(*(int*)arg);

	return 0;
}

int main()
{
	int a = 1;
	void *stack_ptr = malloc(STACK_LENGTH);
	if (stack_ptr == NULL) {
		printf("No virtual memory available\n");
		return -1;
	}
	/*fork();*/

	printf("my pid is %d and address to check is 0x%lx\n", getpid(), (unsigned long) &a);
	clone(run, (stack_ptr + STACK_LENGTH), CLONE_CHILD_CLEARTID|SIGCHLD, &a);
	if (a < 0)
		perror("clone failed. ");

	while (a);

	return 0;
}
```
We are not printing anything after `clone()` to avoid stack overwrite. So we have to assume the child process's `pid` is parent `pid` + 1. This assumption is true most of the times. And we use infinite `while` loop to pause both processes. Loops use `jump` statements. So they will not cause any stack write. So this time we should see same physical address has been used by both processes. Lets see.
``` sh
$ ./specimen
my pid is 3436 and address to check is 0x7fffa920c1ec

```
``` sh
$ sudo ./get_paddr 3436 0x7fffa920c1ec
getting page number of virtual address 140736030884332 of process 3436
opening pagemap /proc/3436/pagemap
moving to 274875060320
physical frame address is 0x67337
physical address is 0x673371ec
$
$ sudo ./get_paddr 3437 0x7fffa920c1ec
getting page number of virtual address 140736030884332 of process 3437
opening pagemap /proc/3437/pagemap
moving to 274875060320
physical frame address is 0x67337
physical address is 0x673371ec
```

GREAT! At last we conquered. We saw the evidence of our beloved *COW*. Now I can sleep peacefully.

### For pure Engineer
Still I feel the urge to make a stack write in child process and see the physical address change. For those curious people out there who want to see the things break, lets run it one more time. Change the `run()` function as follows and execute `specimen.c`.
``` C
int run(void *arg)
{
	*(int*)arg = 10; //stack write
	while(*(int*)arg);

	return 0;
}
```
``` sh
$ ./specimen
my pid is 3549 and address to check is 0x7ffdd8e1e9ac

```
``` sh
$ sudo ./get_paddr 3549 0x7ffdd8e1e9ac
getting page number of virtual address 140728242137516 of process 3549
opening pagemap /proc/3549/pagemap
moving to 274859847920
physical frame address is 0x634e6
physical address is 0x634e69ac
$
$ sudo ./get_paddr 3550 0x7ffdd8e1e9ac
getting page number of virtual address 140728242137516 of process 3550
opening pagemap /proc/3550/pagemap
moving to 274859847920
physical frame address is 0x55976
physical address is 0x559769ac
$
```
Good Nightzzz...
