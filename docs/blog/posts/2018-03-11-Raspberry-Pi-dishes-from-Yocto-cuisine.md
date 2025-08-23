---
title: Raspberry Pi dishes from Yocto cuisine
date: 2018-04-25 21:30:50
thumbnail: "/images/raspberry_pi_3_yocto.jpg"
tags:
 - RaspberryPi
 - yocto
 - build
---

Recently I got some board bring-up work where I come across Yocto project. To complete that project, I had to understand a little more than the basics of Yocto. So I had some hands on like [Yocto Project Quick Start](https://www.yoctoproject.org/docs/2.4.1/yocto-project-qs/yocto-project-qs.html) and [Yocto Project Linux Kernel Development Manual](https://www.yoctoproject.org/docs/2.4.1/kernel-dev/kernel-dev.html). Seems Yocto is so powerful thus I thought of starting to use it for my Raspberry Pi hacking. As expected there are already people tried doing so and there are some good blogs about what they have accomplished. But to my surprise, there is already a meta layer for Raspberry Pi 3. It makes the work even simpler.

I created a directory in Desktop as base for the development. Inside it I created `build`, `downloads`, `sstate`, `tmp` and `yocto` layers directories.
```sh
$ ls
build  downloads  sstate  sstate-cache  tmp  yocto
```
With reference to the [meta-raspberry quick start](http://meta-raspberrypi.readthedocs.io/en/latest/readme.html#quick-start) page, I have cloned [poky](http://git.yoctoproject.org/cgit.cgi/poky), [meta-openembedded](http://git.openembedded.org/meta-openembedded) and [meta-raspberry](https://github.com/agherzan/meta-raspberrypi) inside the `yocto` directory. And checkout to specific branches.
```sh
$ cd yocto
$ git clone git://git.yoctoproject.org/poky.git
$ cd poky
$ git checkout origin/master
$ $ git rev-parse HEAD
da3625c52e1ab8985fba4fc3d133edf92142f182
$
$ cd ..
$ git clone git://git.openembedded.org/meta-openembedded
$ cd meta-openembedded
$ $ git rev-parse HEAD
271ce3576be25d8012af15961f827aa2235e6434
$
$ cd ..
$ git clone git://git.yoctoproject.org/meta-raspberrypi
$ cd meta-raspberry
$ git checkout master
$ git rev-parse HEAD
693f36dded2f29fd77720f7b2e7c88ea76554466
```
You can find all the available layers [here](http://layers.openembedded.org/layerindex/branch/master/layers/)

_Only the above branches and commits worked for me. With rocko branch had pypi.class error_

Sourced the `oe-init` script to setup build environment. And then configured the `conf/bblayers.conf` file to include `meta-layers` from meta-openembedded and meta-raspberry layers. The [meta-raspberry quick start](http://meta-raspberrypi.readthedocs.io/en/latest/readme.html#quick-start) specifies `meta-oe`, `meta-multimedia`, `meta-networking` and `meta-python` as dependencies for `meta-raspberry`. So I'm including all of the into my `conf/bblayers.conf` file.
```sh
$ source yocto/poky/oe-init-build-env build/rpi3/

### Shell environment set up for builds. ###

You can now run 'bitbake <target>'

Common targets are:
    core-image-minimal
    core-image-sato
    meta-toolchain
    meta-ide-support

You can also run generated qemu images with a command like 'runqemu qemux86'
$ pwd
/home/kaba/Desktop/yocto/build/rpi3
$ cat conf/bblayers.conf
# POKY_BBLAYERS_CONF_VERSION is increased each time build/conf/bblayers.conf
# changes incompatibly
POKY_BBLAYERS_CONF_VERSION = "2"

BBPATH = "${TOPDIR}"
BBFILES ?= ""

BBLAYERS ?= " \
  ${TOPDIR}/../../yocto/poky/meta \
  ${TOPDIR}/../../yocto/poky/meta-poky \
  ${TOPDIR}/../../yocto/poky/meta-yocto-bsp \
  ${TOPDIR}/../../yocto/meta-openembedded/meta-oe \
  ${TOPDIR}/../../yocto/meta-openembedded/meta-multimedia \
  ${TOPDIR}/../../yocto/meta-openembedded/meta-networking \
  ${TOPDIR}/../../yocto/meta-openembedded/meta-python \
  ${TOPDIR}/../../yocto/meta-raspberrypi \
  "
$
```
Then made following changes in `conf/local.conf`.
 * Set the `MACHINE` variable to point raspberry Pi 3 [raspberrypi3-64].
 * Set `SSTATE_DIR`, `TMPDIR` and `DL_DIR` to point to `sstate`, `tmp` and `downloads` directories respectively.
 * I usually run my Pi headless. So I enabled `ssh-server-openssh` **EXTRA_IMAGE_FEATURE**. Its up to your choice.
 * Left the other default options as it was.

```sh
$ ls ../../yocto/meta-raspberrypi/conf/machine/
include                 raspberrypi2.conf     raspberrypi-cm3.conf
raspberrypi0.conf       raspberrypi3-64.conf  raspberrypi-cm.conf
raspberrypi0-wifi.conf  raspberrypi3.conf     raspberrypi.conf
$
$ cat conf/local.conf 
MACHINE = "raspberrypi3-64"
DISTRO ?= "poky"
PACKAGE_CLASSES ?= "package_rpm"
EXTRA_IMAGE_FEATURES ?= "debug-tweaks ssh-server-openssh"
USER_CLASSES ?= "buildstats image-mklibs image-prelink"
PATCHRESOLVE = "noop"
CONF_VERSION = "1"

SSTATE_DIR ?= "${TOPDIR}/../../sstate-cache"
TMPDIR ?= "${TOPDIR}/../../tmp"
DL_DIR ?= "${TOPDIR}/../../downloads"

PACKAGECONFIG_append_pn-qemu-native = " sdl"
PACKAGECONFIG_append_pn-nativesdk-qemu = " sdl"
BB_DISKMON_DIRS ??= "\
    STOPTASKS,${TMPDIR},1G,100K \
    STOPTASKS,${DL_DIR},1G,100K \
    STOPTASKS,${SSTATE_DIR},1G,100K \
    STOPTASKS,/tmp,100M,100K \
    ABORT,${TMPDIR},100M,1K \
    ABORT,${DL_DIR},100M,1K \
    ABORT,${SSTATE_DIR},100M,1K \
    ABORT,/tmp,10M,1K"

#SDKMACHINE ?= "i686"
#ASSUME_PROVIDED += "libsdl-native"
$
```
The variable `${TOPDIR}` represents the build directory by default.
`debug-tweaks` will enable root account without password. `ssh-server-openssh` will install an ssh server using openssh. And the ssh server will be started automatically during boot-up.

Build a basic image - `core-image-base`
```sh
$ bitbake core-image-base
WARNING: Layer openembedded-layer should set LAYERSERIES_COMPAT_openembedded-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer multimedia-layer should set LAYERSERIES_COMPAT_multimedia-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer networking-layer should set LAYERSERIES_COMPAT_networking-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer meta-python should set LAYERSERIES_COMPAT_meta-python in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer openembedded-layer should set LAYERSERIES_COMPAT_openembedded-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer multimedia-layer should set LAYERSERIES_COMPAT_multimedia-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer networking-layer should set LAYERSERIES_COMPAT_networking-layer in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Layer meta-python should set LAYERSERIES_COMPAT_meta-python in its conf/layer.conf file to list the core layer names it is compatible with.
WARNING: Host distribution "ubuntu-17.10" has not been validated with this version of the build system; you may possibly experience unexpected failures. It is recommended that you use a tested distribution.
Parsing recipes: 100% |#################################################################################################################################################| Time: 0:03:49
Parsing of 2081 .bb files complete (0 cached, 2081 parsed). 2940 targets, 116 skipped, 0 masked, 0 errors.
NOTE: Resolving any missing task queue dependencies

Build Configuration:
BB_VERSION           = "1.37.0"
BUILD_SYS            = "x86_64-linux"
NATIVELSBSTRING      = "ubuntu-17.10"
TARGET_SYS           = "aarch64-poky-linux"
MACHINE              = "raspberrypi3-64"
DISTRO               = "poky"
DISTRO_VERSION       = "2.5"
TUNE_FEATURES        = "aarch64"
TARGET_FPU           = ""
meta                 
meta-poky            
meta-yocto-bsp       = "HEAD:da3625c52e1ab8985fba4fc3d133edf92142f182"
meta-oe              
meta-multimedia      
meta-networking      
meta-python          = "master:271ce3576be25d8012af15961f827aa2235e6434"
meta-raspberrypi     = "master:693f36dded2f29fd77720f7b2e7c88ea76554466"

NOTE: Fetching uninative binary shim from http://downloads.yoctoproject.org/releases/uninative/1.9/x86_64-nativesdk-libc.tar.bz2;sha256sum=c26622a1f27dbf5b25de986b11584b5c5b2f322d9eb367f705a744f58a5561ec
Initialising tasks: 100% |##############################################################################################################################################| Time: 0:00:04
NOTE: Executing SetScene Tasks
NOTE: Executing RunQueue Tasks
NOTE: Tasks Summary: Attempted 3407 tasks of which 106 didn't need to be rerun and all succeeded.

Summary: There were 9 WARNING messages shown.
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$
```

### Installation and booting
If everything goes well, you can see the final image `tmp/deploy/images/raspberrypi3-64/core-image-base-raspberrypi3-64.rpi-sdimg`. The path is relative to your build directory - *`build/rpi3`*. It is a soft-link to your latest build image. `dd` it to your sd-card - *`/dev/sdb` in my case* - and boot the Raspberry Pi 3 with it.
```sh
$ sudo umount /dev/sdb*
$ sudo dd if=tmp/deploy/images/raspberrypi3-64/core-image-base-raspberrypi3-64.rpi-sdimg of=/dev/sdb bs=1M
$ sudo umount /dev/sdb*
```

### WiFi configuration in Raspberry Pi
To access Raspberry Pi over ssh it should be part of network first. So for the first boot, connect a monitor to the board and boot with your sd-card.

Edit `wpa_supplicant.conf` file to input WiFi access point related information.
```sh
root@raspberrypi3-64:~# cat /etc/wpa_supplicant.conf 
ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=0
update_config=1

network={
	ssid="Bluetooth"
	psk="**********"
}
root@raspberrypi3-64:~#
```
Enter your WiFi access point password against psk.

And bring-up interface `wlan0` on boot-up
```sh
root@raspberrypi3-64:~# cat /etc/init.d/networking
.
.
start)
	.
	.
	ifup wlan0
	.
	.
.
.
```
Configure sticky IP in your access point for your Raspberry Pi corresponding to its mac. So every time after reboot you can straightaway ssh to Raspberry Pi.

### References
 * [https://media.readthedocs.org/pdf/meta-raspberrypi/latest/meta-raspberrypi.pdf]
 * [http://www.jumpnowtek.com/rpi/Raspberry-Pi-Systems-with-Yocto.html]
 * [https://raspinterest.wordpress.com/2016/11/30/yocto-project-on-raspberry-pi-3/]
 * [https://stackoverflow.com/questions/35904769/what-are-the-differences-between-open-embedded-core-and-meta-openembedded]
 * [https://github.com/agherzan/meta-raspberrypi/issues/195]
 * [https://patchwork.openembedded.org/patch/36139/]
