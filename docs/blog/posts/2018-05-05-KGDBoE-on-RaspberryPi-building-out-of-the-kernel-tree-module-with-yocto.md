---
title: KGDBoE on RaspberryPi - building out of the kernel tree module with yocto
date: 2018-05-05 14:43:36
thumbnail: "/images/kgdboe.png"
tags:
 - RaspberryPi
 - yocto
 - kernel module
 - build
---

<div style="color:red;">
<span style="font-weight: bold;">WARNING:</span> KGDBoE cannot be used in Raspberry Pi due to lack of polling support in its network drivers. This post will be useful in future if polling is added. Also this post can be used as a reference for building kernel module that is out of kernel tree.
</div>

### On host 
Building module in Raspberry Pi itself will take more time. Also it requires multiple development tools to be added to final image which will drastically increase the image size. To cross-compile we need SDK for Raspberry Pi.
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ source poky/oe-init-build-env ../build/rpi3/

### Shell environment set up for builds. ###

You can now run 'bitbake <target>'

Common targets are:
    core-image-minimal
    core-image-sato
    meta-toolchain
    meta-ide-support

You can also run generated qemu images with a command like 'runqemu qemux86'
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ bitbake core-image-base -c populate_sdk
WARNING: Host distribution "ubuntu-17.10" has not been validated with this version of the build system; you may possibly experience unexpected failures. It is recommended that you use a tested distribution.
Loading cache: 100% |#####################################################################################################| Time: 0:00:00
Loaded 2783 entries from dependency cache.
NOTE: Resolving any missing task queue dependencies

Build Configuration:
BB_VERSION           = "1.36.0"
BUILD_SYS            = "x86_64-linux"
NATIVELSBSTRING      = "universal"
TARGET_SYS           = "aarch64-poky-linux"
MACHINE              = "raspberrypi3-64"
DISTRO               = "poky"
DISTRO_VERSION       = "2.4.2"
TUNE_FEATURES        = "aarch64"
TARGET_FPU           = ""
meta                 
meta-poky            
meta-yocto-bsp       = "rocko:fdeecc901196bbccd7c5b1ea4268a2cf56764a62"
meta-oe              
meta-multimedia      
meta-networking      
meta-python          = "rocko:a65c1acb1822966c3553de9fc98d8bb6be705c4e"
meta-raspberrypi     = "rocko:20358ec57a8744b0a02da77b283620fb718b0ee0"

Initialising tasks: 100% |################################################################################################| Time: 0:00:11
NOTE: Executing SetScene Tasks
NOTE: Executing RunQueue Tasks
NOTE: Tasks Summary: Attempted 3589 tasks of which 3589 didn't need to be rerun and all succeeded.

Summary: There was 1 WARNING message shown.
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
```
Once done, the script to install SDK will be present in `./tmp/deploy/sdk/`. Run the script `poky-glibc-x86_64-core-image-base-aarch64-toolchain-2.4.2.sh`. This will install the SDK in `/opt/` directory by default.
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ ./tmp/deploy/sdk/poky-glibc-x86_64-core-image-base-aarch64-toolchain-2.4.2.sh 
Poky (Yocto Project Reference Distro) SDK installer version 2.4.2
=================================================================
Enter target directory for SDK (default: /opt/poky/2.4.2): 
The directory "/opt/poky/2.4.2" already contains a SDK for this architecture.
If you continue, existing files will be overwritten! Proceed[y/N]? N
Installation aborted!
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ ls /opt/poky/2.4.2/
environment-setup-aarch64-poky-linux  sysroots/                             
site-config-aarch64-poky-linux        version-aarch64-poky-linux            
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ 
```
I didn't overwrite files as I have installed it already. The script `environment-setup-aarch64-poky-linux` is the one that will setup the cross-build environment for us.

Clone the KGDBoE source from Github. Setup cross-build environment and build the module.
```sh
$ git clone https://github.com/sysprogs/kgdboe.git
Cloning into 'kgdboe'...
remote: Counting objects: 172, done.
remote: Total 172 (delta 0), reused 0 (delta 0), pack-reused 171
Receiving objects: 100% (172/172), 51.95 KiB | 81.00 KiB/s, done.
Resolving deltas: 100% (114/114), done.
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile
$
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile
$ cd kgdboe
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ source /opt/poky/2.4.2/environment-setup-aarch64-poky-linux 
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ make -C /home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build M=$(pwd)
make: Entering directory '/home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build'
  LD      /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/built-in.o
  CC [M]  /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/irqsync.o
  CC [M]  /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_main.o
  CC [M]  /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_io.o
/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_io.c: In function 'force_single_cpu_mode':
/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_io.c:119:3: error: implicit declaration of function 'cpu_down'; did you mean 'cpu_die'? [-Werror=implicit-function-declaration]
   cpu_down(i);
   ^~~~~~~~
   cpu_die
cc1: some warnings being treated as errors
/home/kaba/Desktop/yocto/build/rpi3/tmp/work-shared/raspberrypi3-64/kernel-source/scripts/Makefile.build:293: recipe for target '/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_io.o' failed
make[3]: *** [/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe_io.o] Error 1
/home/kaba/Desktop/yocto/build/rpi3/tmp/work-shared/raspberrypi3-64/kernel-source/Makefile:1493: recipe for target '_module_/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe' failed
make[2]: *** [_module_/home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe] Error 2
Makefile:150: recipe for target 'sub-make' failed
make[1]: *** [sub-make] Error 2
Makefile:24: recipe for target '__sub-make' failed
make: *** [__sub-make] Error 2
make: Leaving directory '/home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build'
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$
```
`cpu_down` function is not available in ARM platform. It is used to shutdown all but one core when debugging, because debugging with single CPU is desirable unless e debug SMP related issues. I couldn't find equivalent ARM function. So let's comment out this call and disable CPUs during boot using `bootargs`.
```
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ git diff
diff --git a/kgdboe_io.c b/kgdboe_io.c
index 1d02360..0d32c36 100644
--- a/kgdboe_io.c
+++ b/kgdboe_io.c
@@ -115,8 +115,8 @@ void force_single_cpu_mode(void)
        printk(KERN_INFO "kgdboe: single-core mode enabled. Shutting down all cores except #0. This is slower, but safer.\n");
        printk(KERN_INFO "kgdboe: you can try using multi-core mode by specifying the following argument:\n");
        printk(KERN_INFO "\tinsmod kgdboe.ko force_single_core = 0\n");
-       for (int i = 1; i < nr_cpu_ids; i++)
-               cpu_down(i);
+       /*for (int i = 1; i < nr_cpu_ids; i++)*/
+               /*cpu_down(i);*/
 }
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$
```
This time the build gave following linker error.
```sh
aarch64-poky-linux-ld: unrecognized option '-Wl,-O1'
```
This is because the `LDFLAGS` environment variable. Kernel module Makefile requires `LDFLAGS` to be set as only the options without `-Wl`. So it appends whatever there in `LDFLAGS` with `-Wl`. As `LDFLAGS` has another `-Wl`, we are getting unrecognized option error. So remove all `-Wl`s from `LDFLAGS`.
```sh
$ echo $LDFLAGS 
-Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed
$ export LDFLAGS="-O1 --hash-style=gnu --as-needed"
$ echo $LDFLAGS 
-O1 --hash-style=gnu --as-needed
```
And build now.
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ make -C /home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build M=$(pwd)
make: Entering directory '/home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build'
  LD [M]  /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe.mod.o
  LD [M]  /home/kaba/Desktop/yocto/build/rpi3/cross-compile/kgdboe/kgdboe.ko
make: Leaving directory '/home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/linux-raspberrypi3_64-standard-build'
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ ls kgdboe.ko
-rw-r--r-- 1 kaba kaba 2.1M May  5 16:11 kgdboe.ko
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$
```
Now `scp` the module `kgdboe.ko` to Raspberry Pi.
```
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3/cross-compile/kgdboe
$ scp kgdboe.ko root@192.168.0.101
```

### On Target
Add `maxcpus=1` option to Raspberry Pi's kernel command line to disable all but one core. The `bootargs` can be found in the file `cmdline.txt` in first partition of the SD card.
```sh
$ ssh root@192.168.0.101
Last login: Sat May  5 07:59:44 2018 from 192.168.0.108
root@raspberrypi3-64:~# ls
kgdboe.ko
root@raspberrypi3-64:~# mount /dev/mmcblk0p1 /boot
root@raspberrypi3-64:~# cat /boot/cmdline.txt 
dwc_otg.lpm_enable=0 console=serial0,115200 root=/dev/mmcblk0p2 rootfstype=ext4 rootwait maxcpus=1
root@raspberrypi3-64:~#
```
Reboot the Pi and try installing the module.
```sh
root@raspberrypi3-64:~# cat /proc/cmdline 
8250.nr_uarts=0 cma=256M bcm2708_fb.fbwidth=1920 bcm2708_fb.fbheight=1080 bcm2708_fb.fbswap=1 vc_mem.mem_base=0x3ec00000 vc_mem.mem_size=0x40000000  dwc_otg.lpm_enable=0 console=ttyS0,115200 root=/dev/mmcblk0p2 rootfstype=ext4 rootwait maxcpus=1
root@raspberrypi3-64:~# insmod kgdboe.ko 
insmod: ERROR: could not insert module kgdboe.ko: Invalid parameters
root@raspberrypi3-64:~# dmesg | tail -5
[ 1211.816691] kgdboe: Failed to setup netpoll for eth0, code -524
[ 8814.348313] netpoll: kgdboe: local IP 1.1.1.101
[ 8814.348344] netpoll: kgdboe: eth0 doesn't support polling, aborting
[ 8814.348360] kgdboe: Failed to setup netpoll for eth0, code -524
root@raspberrypi3-64:~#
```
Due to lack of polling support, Kernel debugging over Ethernet is not available for as of now. If you are reading this, you are part of the clan having hope.

# References
 * [http://sysprogs.com/VisualKernel/kgdboe/tutorial/]
 * [https://github.com/raspberrypi/linux/blob/rpi-4.14.y/arch/arm64/Kconfig]
 * [https://sysprogs.com/w/forums/topic/two-problems-1-linuxkerneldebughelper-seg-fault-and-2-kgdboe-build-fails/]
 * [https://stackoverflow.com/questions/25407320/what-is-the-signification-of-ldflags]
 * [https://github.com/raspberrypi/linux/issues/1877]
