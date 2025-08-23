---
title: Quick kernel upgrade with kexec
date: 2020-10-01 17:30:06
thumbnail: "/images/time_to_upgrade.jpg"
description: "Upgrade kernel of a virtual machine in seconds. Normally kernel upgrade will require a reboot. As reboot will take minutes to complete, service downtime will be more. Here we'll see how upgrade kernel of a VM running Debian-9 in seconds."
tags:
 - linux
 - debian
 - stretch
 - kernel
 - kexec
 - security
 - upgrade
---

One of the major issues we are facing is keeping up to date with security patches. That too keeping the kernel up to date is little harder. Because it requires a reboot. As reboot will take minutes to complete, there will be a significant service downtime. Or doing a service migration to avoid downtime will come with its own complexity.

`kexec` will be help in these situations. It can upgrade the kernel without complete reboot process. Though not zero, the downtime is very less compared to a full reboot. In this post, I'll demo upgrading kernel of a Virtual machine running Debian-9.

This VM is running Debian `Linux-4.9.0-12`. Let me update to the latest kernel available now - `Linux-4.9.0-13`.

Install `kexec-tools`
```sh
root@debian:~# apt install kexec-tools -qq
```

Install latest `Linux-image` package. This will not overwrite the existing kernel or initrd image in your `/boot/` directory. So you can safely rollback if required.
```sh
root@debian:~# ls -lh /boot/
total 25M
-rw-r--r-- 1 root root 3.1M Jan 21  2020 System.map-4.9.0-12-amd64
-rw-r--r-- 1 root root 183K Jan 21  2020 config-4.9.0-12-amd64
drwxr-xr-x 5 root root 4.0K Apr 24 12:40 grub
-rw-r--r-- 1 root root  18M Apr 24 12:23 initrd.img-4.9.0-12-amd64
-rw-r--r-- 1 root root 4.1M Jan 21  2020 vmlinuz-4.9.0-12-amd64

root@debian:~# sudo apt update -qq
43 packages can be upgraded. Run 'apt list --upgradable' to see them.

root@debian:~# sudo apt install linux-image-amd64 -qq
The following additional packages will be installed:
  linux-image-4.9.0-13-amd64
Suggested packages:
  linux-doc-4.9 debian-kernel-handbook
The following NEW packages will be installed:
  linux-image-4.9.0-13-amd64
The following packages will be upgraded:
  linux-image-amd64
1 upgraded, 1 newly installed, 0 to remove and 42 not upgraded.
Need to get 39.3 MB of archives.
After this operation, 193 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
Selecting previously unselected package linux-image-4.9.0-13-amd64.
(Reading database ... 26429 files and directories currently installed.)
Preparing to unpack .../linux-image-4.9.0-13-amd64_4.9.228-1_amd64.deb ...
Unpacking linux-image-4.9.0-13-amd64 (4.9.228-1) ...........................]
Preparing to unpack .../linux-image-amd64_4.9+80+deb9u11_amd64.deb .........]
Unpacking linux-image-amd64 (4.9+80+deb9u11) over (4.9+80+deb9u10) .........]
Setting up linux-image-4.9.0-13-amd64 (4.9.228-1) ..........................]
I: /vmlinuz is now a symlink to boot/vmlinuz-4.9.0-13-amd64.................]
I: /initrd.img is now a symlink to boot/initrd.img-4.9.0-13-amd64
/etc/kernel/postinst.d/initramfs-tools:
update-initramfs: Generating /boot/initrd.img-4.9.0-13-amd64
/etc/kernel/postinst.d/zz-update-grub:
Generating grub configuration file ...
Found linux image: /boot/vmlinuz-4.9.0-13-amd64
Found initrd image: /boot/initrd.img-4.9.0-13-amd64
Found linux image: /boot/vmlinuz-4.9.0-12-amd64
Found initrd image: /boot/initrd.img-4.9.0-12-amd64
done
Setting up linux-image-amd64 (4.9+80+deb9u11) ...###########................]

root@debian:~# ls -lh /boot/
total 50M
-rw-r--r-- 1 root root 3.1M Jan 21  2020 System.map-4.9.0-12-amd64
-rw-r--r-- 1 root root 3.1M Jul  6 02:59 System.map-4.9.0-13-amd64   <---
-rw-r--r-- 1 root root 183K Jan 21  2020 config-4.9.0-12-amd64
-rw-r--r-- 1 root root 183K Jul  6 02:59 config-4.9.0-13-amd64       <---
drwxr-xr-x 5 root root 4.0K Oct  1 17:25 grub
-rw-r--r-- 1 root root  18M Apr 24 12:23 initrd.img-4.9.0-12-amd64
-rw-r--r-- 1 root root  18M Oct  1 17:25 initrd.img-4.9.0-13-amd64   <---
-rw-r--r-- 1 root root 4.1M Jan 21  2020 vmlinuz-4.9.0-12-amd64
-rw-r--r-- 1 root root 4.1M Jul  6 02:59 vmlinuz-4.9.0-13-amd64      <---
```

Now copy the kernel command line from `/proc/cmdline`. We should pass this to `kexec`.
```sh
root@debian:~# cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-4.9.0-12-amd64 root=UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx ro net.ifnames=0 biosdevname=0 cgroup_enable=memory console=tty0 console=ttyS0,115200 notsc scsi_mod.use_blk_mq=Y quiet
```

Load the new kernel using `kexec -l`.
```sh
root@debian:~# kexec -l /boot/vmlinuz-4.9.0-13-amd64 --initrd=/boot/initrd.img-4.9.0-13-amd64 --command-line="BOOT_IMAGE=/boot/vmlinuz-4.9.0-13-amd64 root=UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx ro net.ifnames=0 biosdevname=0 cgroup_enable=memory console=tty0 console=ttyS0,115200 notsc scsi_mod.use_blk_mq=Y quiet"
root@debian:~#
```

Now upgrade to the new kernel.
```sh
root@debian:~# uname -a
Linux debian 4.9.0-12-amd64 #1 SMP Debian 4.9.210-1 (2020-01-20) x86_64 GNU/Linux

root@debian:~# systemctl start kexec.target
[268181.341191] kexec_core: Starting new kernel
/dev/sda1: clean, 35704/655360 files, 366185/2621179 blocks
GROWROOT: NOCHANGE: partition 1 is size 20969439. it cannot be grown

Debian GNU/Linux 9 debian ttyS0

debian login: root
Password:
Last login: Mon Sep 28 14:59:30 IST 2020 on ttyS0
Linux debian 4.9.0-13-amd64 #1 SMP Debian 4.9.228-1 (2020-07-05) x86_64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.

root@debian:~# uname -a
Linux debian 4.9.0-13-amd64 #1 SMP Debian 4.9.228-1 (2020-07-05) x86_64 GNU/Linux
root@debian:~#
```

## Time to upgrade
This actually took no time. I was pinging this VM from its Host. There was a slight increase in latency while the upgrade was in progress. That was less than a second. But I didn't run any service and tested its status after reboot. Because it may vary from service to service.

```sh
64 bytes from 192.168.122.91: icmp_seq=176 ttl=64 time=0.465 ms
64 bytes from 192.168.122.91: icmp_seq=177 ttl=64 time=0.408 ms
64 bytes from 192.168.122.91: icmp_seq=181 ttl=64 time=8.32 ms   <---
64 bytes from 192.168.122.91: icmp_seq=182 ttl=64 time=0.452 ms
64 bytes from 192.168.122.91: icmp_seq=183 ttl=64 time=0.198 ms

```
