---
title: Debugging application with cross-GDB in Yocto environment
date: 2018-06-03 11:51:43
thumbnail: "/images/gdb_tui_screenshot.png"
tags:
 - gdb
 - debug
 - yocto
---

GDB is a very useful tool when debugging applications. Embedded devices lack sophisticated features that Development machines enjoy like better processor speed, huge RAM, etc. In such cases running GDB on the target will be painfully slow. But thanks to remote-debugging support from GDB which saves us from such situations. In this post we'll see how to do remote debugging of an application running on Raspberry Pi 3.

You can read previous posts to have better understanding about [Yocto build environment](http://eastrivervillage.com/Raspberry-Pi-dishes-from-Yocto-cuisine/) and how to [setup serial connection with Raspberry pi 3](https://eastrivervillage.com/KGDB-KDB-over-serial-with-RaspberryPi/).

### Target program and compilation
Going along with the world, we'll start with traditional `Hello World` program. Write the C program and cross-compile it to `arm64` platform. Secure copy the executable binary to Raspberry Pi 3.
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
$ source /opt/poky/2.4.2/environment-setup-aarch64-poky-linux 
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ $CC -g syscall.c -o syscall.arm64.debug
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ $CC syscall.c -o syscall.arm64
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ ls
total 36K
-rwxr-xr-x 1 kaba kaba 14K Jun  3 12:07 syscall.arm64
-rwxr-xr-x 1 kaba kaba 16K Jun  3 12:07 syscall.arm64.debug
-rw-r--r-- 1 kaba kaba  73 May 31 06:48 syscall.c
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ scp ./syscall.arm64 root@192.168.0.101:
syscall.arm64                                                                                                                                                             100%   13KB 488.8KB/s   00:00    

kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ 
```
We have two binaries compiled here. We have copied the one without symbols to target board. And the one with debug symbols will be passed as an argument `cross-GDB`.

### Build GDB-server
Add `gdbserver` tool to the target image by enabling `tools-debug` in `EXTRA_IMAGE_FEATURES`. Run the `Hello World` program in target, attached with `gdbserver` that listens on network.
```sh
root@raspberrypi3-64:~# gdbserver 192.168.0.101:2345 ./syscall.arm64 
Process ./syscall.arm64 created; pid = 1671
Listening on port 2345


```
It will start the program and wait for `remote-gdb` to attach.

### Build GDB with TUI support
`GDB` has a `tui` [Text User Interface] mode which will be very much useful while debugging. With `tui` enabled, we can see the code that runs, equivalent assembly instruction, current register values, etc., simultaneously. But by default the `GDB` in Yocto doesn't build with `tui` support. Append `--enable-tui` option to `gdb-cross` bbfile. And build the SDK as mentioned [here](https://eastrivervillage.com/KGDBoE-on-RaspberryPi-building-out-of-the-kernel-tree-module-with-yocto/).
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ tree meta-kaba-hacks/
meta-kaba-hacks/
├── conf
│   └── layer.conf
├── COPYING.MIT
├── recipes-devtools
│   └── gdb
│       └── gdb-%.bbappend
└── recipes-kernel
    └── linux
        ├── linux-raspberrypi
        │   ├── debug.cfg
        │   └── enable_proc_zconfig.cfg
        └── linux-raspberrypi_4.9.bbappend

kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ cat meta-kaba-hacks/recipes-devtools/gdb/gdb-%.bbappend 
EXTRA_OECONF += " --enable-tui"
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ bitbake core-image-base -c populate_sdk
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ ./tmp/deploy/sdk/poky-glibc-x86_64-core-image-base-aarch64-toolchain-2.4.2.sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ 
```

### Launch
Run `cross-GDB` with `tui` enabled and connect it to the target.
```sh
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ source /opt/poky/2.4.2/environment-setup-aarch64-poky-linux 
kaba@kaba-Vostro-1550:~/Desktop/workbench/code
$ $GDB -tui ./syscall.arm64.debug 
```
`GDB` window will open-up similar to the screenshot at the top of this post (zoom-in your web page to see the image details clearly). Below I'm copying only the command-panel of `GDB`. Based on the commands you run, the other panels will also get changed.

Connect to target program waiting for remote-GDB.
```sh
(gdb) target remote 192.168.0.101:2345
Remote debugging using 192.168.0.101:2345
Reading /lib/ld-linux-aarch64.so.1 from remote target...
warning: File transfers from remote targets can be slow. Use "set sysroot" to access files locally instead.
Reading /lib/ld-linux-aarch64.so.1 from remote target...
Reading symbols from target:/lib/ld-linux-aarch64.so.1...Reading /lib/ld-2.26.so from remote target...
Reading /lib/.debug/ld-2.26.so from remote target...
(no debugging symbols found)...done.
0x0000007fb7fd2f40 in ?? () from target:/lib/ld-linux-aarch64.so.1
(gdb) 
```

Now it connected to the target program. But it is not very much helpful. Because it tries to load the symbols and source from target board. The target has no source or symbol. Lets direct `GDB` to load source and symbol from host machine.
```sh
(gdb) set sysroot /opt/poky/2.4.2/sysroots/aarch64-poky-linux/
warning: .dynamic section for "/opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/ld-linux-aarch64.so.1" is not at the expected address (wrong library or version mismatch?)
Reading symbols from /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/ld-linux-aarch64.so.1...Reading symbols from /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/.debug/ld-2.26.so...done.
done.
Reading symbols from /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/ld-linux-aarch64.so.1...Reading symbols from /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/.debug/ld-2.26.so...done.
done.
(gdb) info sharedlibrary  
From                To                  Syms Read   Shared Object Library
0x0000007fb7fd2dc0  0x0000007fb7fe9fc8  Yes         /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/ld-linux-aarch64.so.1
(gdb)
```
Set a break-point in `main` and continue. So the program will pause once it reaches main. Now if you see, there will be another shared library loaded.
```sh
(gdb) break main
Breakpoint 1 at 0x40056c: file syscall.c, line 5.
(gdb) c
Continuing.

Breakpoint 1, main () at syscall.c:5
(gdb) info sharedlibrary
From                To                  Syms Read   Shared Object Library
0x0000007fb7fd2dc0  0x0000007fb7fe9fc8  Yes         /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/ld-linux-aarch64.so.1
0x0000007fb7ea0700  0x0000007fb7f8a118  Yes         /opt/poky/2.4.2/sysroots/aarch64-poky-linux/lib/libc.so.6
(gdb) 
```
Now you can see the line `printf` on the top panel will be highlighted. It means, that is the current line to be executed. Lets step in to the definition of `printf`.
```sh
(gdb) s
_IO_puts (str=0x400638 "Hello world!") at /usr/src/debug/glibc/2.26-r0/git/libio/ioputs.c:36
(gdb)
```
The control went into the library function. But we can see the top panel starts showing `No Source Available` error. This is because `GDB` searches the source files in wrong directory. Have a second look at last output. `GDB` searches `ioputs.c` in `/usr/src/debug/glibc/2.26-r0/git/libio` directory. Lets set the path to shared library correct.
```sh
(gdb) set substitute-path /usr/src/debug/ /home/kaba/Desktop/yocto/build/rpi3/tmp/work/aarch64-poky-linux/
(gdb)
```
Now you can see the source started appearing in the top-panel. Read more about `GDB` commands and have fun.

# References
 * [https://sourceware.org/gdb/onlinedocs/gdb/Source-Path.html]
 * [http://visualgdb.com/gdbreference/commands/sharedlibrary]
 * [http://visualgdb.com/gdbreference/commands/set_solib-search-path]
 * [https://www.yoctoproject.org/docs/1.4.2/adt-manual/adt-manual.html]
 * [https://www.yoctoproject.org/docs/latest/mega-manual/mega-manual.html]
