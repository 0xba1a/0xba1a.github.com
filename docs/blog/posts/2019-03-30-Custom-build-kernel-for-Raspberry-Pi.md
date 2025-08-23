---
title: Custom build kernel for Raspberry Pi
date: 2019-03-30 18:26:30
thumbnail: "/images/custom_kernel_rpi3.jpg"
description: "How to build your own customized kernel for Raspberry Pi using yocto."
author:
 - Bala
tags:
 - raspberrypi
 - linux
 - kernel
---

I've already written a post about how to [cross-compile mainline kernel for Raspberry Pi](2018-10-06-64-bit-Mainline-kernel-on-Raspberry-Pi-3.md). In this post I'm covering how to cross-compile [Raspberry Pi Linux](https://github.com/raspberrypi/linux). This will be simple and straight forward. I may write a series of posts related to kernel debugging, optimization which will be based on Raspberry Pi kernel. So this post will be starting point for them.

Directory structure,
```sh
balakumaran@balakumaran-pc:~/Desktop/RPi$ ls -lh
total 32K
drwxrwxr-x  3 balakumaran balakumaran 4.0K Mar  9 19:33 firmware
drwxr-xr-x  8 balakumaran balakumaran 4.0K Jan 23 01:52 gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu
drwxrwxr-x 22 balakumaran balakumaran 4.0K Mar 30 18:38 kernel_out
drwxrwxr-x 26 balakumaran balakumaran 4.0K Mar 30 18:13 linux-rpi-4.14.y
drwxrwxr-x 18 balakumaran balakumaran 4.0K Mar  9 19:34 rootfs
balakumaran@balakumaran-pc:~/Desktop/RPi$
```
Directory		| Purpose													|
----------------|-----------------------------------------------------------|
gcc-li...		| GCC cross compiler from Linaro. Extracted					|
firmware/boot	| boot directory of Raspberry Pi firmware repo				|
kernel_out		| Output directory for Raspberry kernel						|
rootfs			| rootfs from Linaro. Extracted								|
linux-rpi...	| Raspberry Pi kernel repo									|

Used [Ubuntu image rootfs](https://releases.linaro.org/archive/ubuntu/images/developer-arm64/) from [Linaro](https://linaro.org).

## Prepare SD card
Make two partition as follows,
```sh
balakumaran@balakumaran-pc:~/Desktop/RPi$ sudo fdisk -l /dev/sdc
[sudo] password for balakumaran:
Disk /dev/sdc: 29.7 GiB, 31914983424 bytes, 62333952 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0xcde890ba

Device     Boot  Start      End  Sectors  Size Id Type
/dev/sdc1  *      2048   133119   131072   64M  b W95 FAT32
/dev/sdc2       133120 62333951 62200832 29.7G 83 Linux
balakumaran@balakumaran-pc:~/Desktop/RPi$
```
Complete steps on how to do this is available in Appendix.

Copy necessary files
```sh
balakumaran@balakumaran-pc:/media/balakumaran$ sudo mount /dev/sdc1 /mnt/boot/
balakumaran@balakumaran-pc:/media/balakumaran$ sudo mount /dev/sdc2 /mnt/rootfs/
balakumaran@balakumaran-pc:~/Desktop$ sudo cp -rf ~/Desktop/RPi/firmware/boot/* /mnt/boot/
[sudo] password for balakumaran:
balakumaran@balakumaran-pc:~/Desktop$
balakumaran@balakumaran-pc:~/Desktop$ sudo cp -rf ~/Desktop/RPi/rootfs/* /mnt/rootfs/
balakumaran@balakumaran-pc:~/Desktop$
```

## Build and Install kernel
Unless you are ready for the pain, use stable kernel release.

Setup following environmental variables,
```sh
balakumaran@balakumaran-pc:~/Desktop/RPi$ source ~/setup_arm64_build.sh
balakumaran@balakumaran-pc:~/Desktop/RPi$ echo $CROSS_COMPILE
aarch64-linux-gnu-
balakumaran@balakumaran-pc:~/Desktop/RPi$ echo $ARCH
arm64
balakumaran@balakumaran-pc:~/Desktop/RPi$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/balakumaran/Desktop/RPi/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/
balakumaran@balakumaran-pc:~/Desktop/RPi$
```

Cross compile Kernel, Device-tree, modules.
```sh
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ time make ARCH=arm64 O=../kernel_out/ bcmrpi3_defconfig
make[1]: Entering directory '/home/balakumaran/Desktop/RPi/kernel_out'
.
.
.

balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ time make -j8  ARCH=arm64 O=../kernel_out/
make[1]: Entering directory '/home/balakumaran/Desktop/RPi/kernel_out'
.
.
.

balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ make ARCH=arm64 O=../kernel_out/ dtbs
make[1]: Entering directory '/home/balakumaran/Desktop/RPi/kernel_out'
.
.
.

balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ sudo cp ../kernel_out/arch/arm64/boot/Image /mnt/boot/kernel8.img
[sudo] password for balakumaran:
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ sudo make  ARCH=arm64 O=../kernel_out/ INSTALL_PATH=/mnt/boot/ dtbs_install
make[1]: Entering directory '/home/balakumaran/Desktop/RPi/kernel_out'
arch/arm64/Makefile:27: ld does not support --fix-cortex-a53-843419; kernel may be susceptible to erratum
arch/arm64/Makefile:40: LSE atomics not supported by binutils
arch/arm64/Makefile:48: Detected assembler with broken .inst; disassembly will be unreliable
make[3]: Nothing to be done for '__dtbs_install'.
  INSTALL arch/arm64/boot/dts/al/alpine-v2-evp.dtb
.
.
.
```

Create `cmdline.txt` and `config.txt`.
```sh
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ cat /mnt/boot/cmdline.txt
dwc_otg.lpm_enable=0 console=serial0,115200 root=/dev/mmcblk0p2 rootfstype=ext4 rootwait
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ cat /mnt/boot/config.txt
dtoverlay=pi3-disable-bt
disable_overscan=1
dtparam=audio=on
device_tree=dtbs/4.14.98-v8+/broadcom/bcm2710-rpi-3-b.dtb
overlay_prefix=dtbs/4.14.98-v8+/overlays/
enable_uart=1
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$
```

## Prepare rootfs
I'm going to use ubuntu-base images with some additional modification as rootfs here. Find ubuntu-base releases at [http://cdimage.ubuntu.com/ubuntu-base/releases/](http://cdimage.ubuntu.com/ubuntu-base/releases/). Latest stable is always better. Download and extract ubuntu-base rootfs. Install kernel modules into the rootfs extracted.

```sh
balakumaran@balakumaran-pc:~/Desktop/RPi/linux-rpi-4.14.y$ sudo make  ARCH=arm64 O=../kernel_out/ INSTALL_MOD_PATH=$HOME/ubuntu-base/ modules_install
[sudo] password for balakumaran:
make[1]: Entering directory '/home/balakumaran/Desktop/RPi/kernel_out'
arch/arm64/Makefile:27: ld does not support --fix-cortex-a53-843419; kernel may be susceptible to erratum
arch/arm64/Makefile:40: LSE atomics not supported by binutils
arch/arm64/Makefile:48: Detected assembler with broken .inst; disassembly will be unreliable
  INSTALL arch/arm64/crypto/aes-neon-blk.ko
.
.
.

```

Copy your resolv.conf for network access.
```sh
$ sudo cp -av /run/systemd/resolve/stub-resolv.conf $HOME/rootfs/etc/resolv.conf
```

Lets `chroot` into the new rootfs and install necessary packages. But its an arm64 rootfs. So you need `qemu` user-mode emulation. Install `qemu-user-static` in your host Ubuntu and copy that to new rootfs. And then chroot will work.
```sh
$ sudo apt install qemu-user-static
.
.
.

$ sudo cp /usr/bin/qemu-aarch64-static $HOME/rootfs/usr/bin/
$ sudo chroot $HOME/rootfs/
```

Change `root` user password and install necessary packages. As these binaries are running on emulator, they will be bit slower. Its just one time.
```sh
$ passwd root
$ apt-get update
$ apt-get upgrade
$ apt-get install sudo ifupdown net-tools ethtool udev wireless-tools iputils-ping resolvconf wget apt-utils wpasupplicant kmod systemd vim
```

<strong>NOTE:</strong>
If you face any error like `cannot create key file at /tmp/`, change permission of `tmp`.
```sh
$ chmod 777 /tmp
```

Download raspberry firmware-nonfree package from [raspberry repository](https://archive.raspberrypi.org/debian/pool/main/f/firmware-nonfree/firmware-brcm80211_20161130-3+rpt3_all.deb), extract wireless firmware and copy it to rootfs. Refer this [answer](https://www.linuxquestions.org/questions/slackware-arm-108/raspberry-pi-3-b-wifi-nic-not-found-4175627137/#post5840054) for more details. As I'm having a RPI3b board, I copied `brcmfmac43430-sdio.bin` and `brcmfmac43430-sdio.txt` to `lib/firmware/brcm`
```sh
$ mkdir -p $HOME/lib/modules/brcm/
$ cp brcmfmac43430-sdio.txt brcmfmac43430-sdio.bin $HOME/lib/modules/brcm/
```

Edit `etc/fstab` or rootfs will be mounted as read-only.
```sh
echo "/dev/mmcblk0p2	/	ext4	defaults,noatime	0	1" >> $HOME/rootfs/etc/fstab
```
I referred [this](https://a-delacruz.github.io/ubuntu/rpi3-setup-filesystem.html) link for rootfs preparation. Though I'm not using, there are steps to remove unwanted files explained.

## Reference
 * https://a-delacruz.github.io/ubuntu/rpi3-setup-64bit-kernel
 * https://a-delacruz.github.io/ubuntu/rpi3-setup-filesystem.html
 * https://www.linuxquestions.org/questions/slackware-arm-108/raspberry-pi-3-b-wifi-nic-not-found-4175627137/#post5840054
 * http://cdimage.ubuntu.com/ubuntu-base/releases/
 * https://raspberrypi.stackexchange.com/questions/61319/how-to-add-wifi-drivers-in-custom-kernel


## Appendix
```sh
Command (m for help): p
Disk /dev/sdc: 29.7 GiB, 31914983424 bytes, 62333952 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0xcde890ba

Device     Boot  Start      End  Sectors  Size Id Type
/dev/sdc1  *      2048   133119   131072   64M  b W95 FAT32
/dev/sdc2       133120 62333951 62200832 29.7G 83 Linux

Command (m for help): d
Partition number (1,2, default 2): 2

Partition 2 has been deleted.

Command (m for help): d
Selected partition 1
Partition 1 has been deleted.

Command (m for help): p
Disk /dev/sdc: 29.7 GiB, 31914983424 bytes, 62333952 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0xcde890ba

Command (m for help): n
Partition type
   p   primary (0 primary, 0 extended, 4 free)
   e   extended (container for logical partitions)
Select (default p): p
Partition number (1-4, default 1):
First sector (2048-62333951, default 2048):
Last sector, +sectors or +size{K,M,G,T,P} (2048-62333951, default 62333951): +64M

Created a new partition 1 of type 'Linux' and of size 64 MiB.
Partition #1 contains a vfat signature.

Do you want to remove the signature? [Y]es/[N]o: Y

The signature will be removed by a write command.

Command (m for help): n
Partition type
   p   primary (1 primary, 0 extended, 3 free)
   e   extended (container for logical partitions)
Select (default p): p
Partition number (2-4, default 2):
First sector (133120-62333951, default 133120):
Last sector, +sectors or +size{K,M,G,T,P} (133120-62333951, default 62333951):

Created a new partition 2 of type 'Linux' and of size 29.7 GiB.
Partition #2 contains a ext4 signature.

Do you want to remove the signature? [Y]es/[N]o: Y

The signature will be removed by a write command.

Command (m for help): t
Partition number (1,2, default 2): 1
Hex code (type L to list all codes): L

 0  Empty           24  NEC DOS         81  Minix / old Lin bf  Solaris
 1  FAT12           27  Hidden NTFS Win 82  Linux swap / So c1  DRDOS/sec (FAT-
 2  XENIX root      39  Plan 9          83  Linux           c4  DRDOS/sec (FAT-
 3  XENIX usr       3c  PartitionMagic  84  OS/2 hidden or  c6  DRDOS/sec (FAT-
 4  FAT16 <32M      40  Venix 80286     85  Linux extended  c7  Syrinx
 5  Extended        41  PPC PReP Boot   86  NTFS volume set da  Non-FS data
 6  FAT16           42  SFS             87  NTFS volume set db  CP/M / CTOS / .
 7  HPFS/NTFS/exFAT 4d  QNX4.x          88  Linux plaintext de  Dell Utility
 8  AIX             4e  QNX4.x 2nd part 8e  Linux LVM       df  BootIt
 9  AIX bootable    4f  QNX4.x 3rd part 93  Amoeba          e1  DOS access
 a  OS/2 Boot Manag 50  OnTrack DM      94  Amoeba BBT      e3  DOS R/O
 b  W95 FAT32       51  OnTrack DM6 Aux 9f  BSD/OS          e4  SpeedStor
 c  W95 FAT32 (LBA) 52  CP/M            a0  IBM Thinkpad hi ea  Rufus alignment
 e  W95 FAT16 (LBA) 53  OnTrack DM6 Aux a5  FreeBSD         eb  BeOS fs
 f  W95 Ext'd (LBA) 54  OnTrackDM6      a6  OpenBSD         ee  GPT
10  OPUS            55  EZ-Drive        a7  NeXTSTEP        ef  EFI (FAT-12/16/
11  Hidden FAT12    56  Golden Bow      a8  Darwin UFS      f0  Linux/PA-RISC b
12  Compaq diagnost 5c  Priam Edisk     a9  NetBSD          f1  SpeedStor
14  Hidden FAT16 <3 61  SpeedStor       ab  Darwin boot     f4  SpeedStor
16  Hidden FAT16    63  GNU HURD or Sys af  HFS / HFS+      f2  DOS secondary
17  Hidden HPFS/NTF 64  Novell Netware  b7  BSDI fs         fb  VMware VMFS
18  AST SmartSleep  65  Novell Netware  b8  BSDI swap       fc  VMware VMKCORE
1b  Hidden W95 FAT3 70  DiskSecure Mult bb  Boot Wizard hid fd  Linux raid auto
1c  Hidden W95 FAT3 75  PC/IX           bc  Acronis FAT32 L fe  LANstep
1e  Hidden W95 FAT1 80  Old Minix       be  Solaris boot    ff  BBT
Hex code (type L to list all codes): b

Changed type of partition 'Linux' to 'W95 FAT32'.

Command (m for help): t
Partition number (1,2, default 2): 2
Hex code (type L to list all codes): 83

Changed type of partition 'Linux' to 'Linux'.

Command (m for help): w
The partition table has been altered.
Calling ioctl() to re-read partition table.
Syncing disks.

balakumaran@balakumaran-pc:/media/balakumaran$
balakumaran@balakumaran-pc:/media/balakumaran$ sudo fdisk -l /dev/sdc
Disk /dev/sdc: 29.7 GiB, 31914983424 bytes, 62333952 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0xcde890ba

Device     Boot  Start      End  Sectors  Size Id Type
/dev/sdc1         2048   133119   131072   64M  b W95 FAT32
/dev/sdc2       133120 62333951 62200832 29.7G 83 Linux
balakumaran@balakumaran-pc:/media/balakumaran$
balakumaran@balakumaran-pc:/media/balakumaran$ sudo mkfs.fat /dev/sdc1
mkfs.fat 4.1 (2017-01-24)
balakumaran@balakumaran-pc:/media/balakumaran$ sudo mkfs.ext4 /dev/sdc2
mke2fs 1.44.4 (18-Aug-2018)
Creating filesystem with 7775104 4k blocks and 1945888 inodes
Filesystem UUID: 5815d093-6381-4db7-b692-32192b24cf9c
Superblock backups stored on blocks:
        32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208,
        4096000

Allocating group tables: done
Writing inode tables: done
Creating journal (32768 blocks): done
Writing superblocks and filesystem accounting information: done

balakumaran@balakumaran-pc:/media/balakumaran$
balakumaran@balakumaran-pc:/media/balakumaran$ sudo fdisk /dev/sdc

Welcome to fdisk (util-linux 2.32).
Changes will remain in memory only, until you decide to write them.
Be careful before using the write command.


Command (m for help): a
Partition number (1,2, default 2): 1

The bootable flag on partition 1 is enabled now.

Command (m for help): p
Disk /dev/sdc: 29.7 GiB, 31914983424 bytes, 62333952 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0xcde890ba

Device     Boot  Start      End  Sectors  Size Id Type
/dev/sdc1  *      2048   133119   131072   64M  b W95 FAT32
/dev/sdc2       133120 62333951 62200832 29.7G 83 Linux

Command (m for help): w
The partition table has been altered.
Calling ioctl() to re-read partition table.
Syncing disks.

balakumaran@balakumaran-pc:/media/balakumaran$
```
