---
title: KGDB/KDB over serial with Raspberry Pi
date: 2018-05-23 19:47:53
thumbnail: "/images/raspberry_pi_pl2303_serial_2.jpg"
authors:
 - Bala
tags:
 - RaspberryPi
 - yocto
 - kernel
 - debug
---

### Hardware setup
We need a USB to serial converter to connect Raspberry Pi to the PC serially. [This](https://www.amazon.in/PL2303-Converter-Adapter-Aurdino-Raspberry/dp/B00UZERG94/ref=sr_1_1?ie=UTF8&qid=1527085291&sr=8-1&keywords=pl2303+usb+to+rs232) is the cheapest among all converters available in Amazon. This is based on PL2302 chip. I'm not sure it's original or Chinese replica. In my case it worked out of the box with Ubuntu-17.10. In case if it throws error-10, try downgrading your PL2303 driver. Because the manufacturer blocked all counterfeit chips in his latest driver. I ordered this one and a set of [female-to-female jumper wires](https://www.amazon.in/Jumper-Wires-Male-female-Pieces/dp/B00ZYFX6A2/ref=sr_1_4?s=industrial&ie=UTF8&qid=1527085464&sr=1-4&keywords=jumper+wires). Wait for three days and continue with this article.

<br />
Interfacing is simple
- Connect 5V to 5V
- Connect TX of converter with RxD of Raspberry's GPIO UART
- Connect RX of converter with TxD of Raspberry's GPIO UART
- Connect the ground to ground

<br />
Below the Raspberry Pi GPIO pin layout
![Raspberry Pi 3 GPIO pin layout](images/raspberry_pi_3_pin_layout.jpg)
<br />
PL2303's pin layout
![PL2303 USB to TTL converter pin layout](images/pl2303_pin_layout.jpg)
<br />
And the connection goes like this,
![Raspberry Pi 3 serial connection](images/raspberry_pi_pl2303_serial_1.jpg)
![Raspberry Pi 3 serial connection](images/raspberry_pi_pl2303_serial_2.jpg)
<br />
As power is supplied via GPIO pin, there is no need of external power supply. I don't know what will happen if both power sources are connected. I'm not dare to try.

### Software setup
- Kernel has to be built with debug support
- Enable `tui` support for GDB - if required

For that I have created a custom layer with following tree structure.
```sh
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
```
Enable debug symbols and KGDB/KDB related configs in Linux.
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ cat meta-kaba-hacks/recipes-kernel/linux/linux-raspberrypi_4.9.bbappend
FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"
SRC_URI += "\
			file://debug.cfg \
			file://enable_proc_zconfig.cfg \
			"
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ cat meta-kaba-hacks/recipes-kernel/linux/linux-raspberrypi/debug.cfg
# CONFIG_STRICT_KERNEL_RWX is not set
CONFIG_DEBUG_INFO=y
CONFIG_FRAME_POINTER=y
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y
CONFIG_KGDB_KDB=y
CONFIG_KDB_KEYBOARD=y
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ cat meta-kaba-hacks/recipes-kernel/linux/linux-raspberrypi/enable_proc_zconfig.cfg
CONFIG_IKCONFIG=y
CONFIG_IKCONFIG_PROC=y
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$
```
By default `tui` options is disabled for `GDB` in yocto. So I'm overriding it to enable it.
```
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$ cat meta-kaba-hacks/recipes-devtools/gdb/gdb-%.bbappend
EXTRA_OECONF += " --enable-tui"
kaba@kaba-Vostro-1550:~/Desktop/yocto/yocto
$
```

Build `kernel` and `populate_sdk`. Refer [this](http://eastrivervillage.com/KGDBoE-on-RaspberryPi-building-out-of-the-kernel-tree-module-with-yocto/) and [this](http://eastrivervillage.com/Raspberry-Pi-dishes-from-Yocto-cuisine/) if you need help on working with Yocto. And copy the newly built image to Raspberry Pi.

### Enable Serial in Raspberry Pi
By default serial interface is not enabled in yocto built Raspberry Pi distribution. We have to enable it in the config.txt file. Connect the SD-card written with Raspberry Pi image to PC and mount first partition. Append `enable_uart=1` to the config.txt file.
```sh
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ mount | grep sdb1
/dev/sdb1 on /media/kaba/raspberrypi type vfat (rw,nosuid,nodev,relatime,uid=1000,gid=1000,fmask=0022,dmask=0022,codepage=437,iocharset=iso8859-1,shortname=mixed,showexec,utf8,flush,errors=remount-ro,uhelper=udisks2)
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$ tail /media/kaba/raspberrypi/config.txt
#dtparam=pwr_led_gpio=35
# Enable VC4 Graphics
dtoverlay=vc4-fkms-v3d,cma-256
# have a properly sized image
disable_overscan=1
# Enable audio (loads snd_bcm2835)
dtparam=audio=on
# Load correct Device Tree for Aarch64
device_tree=bcm2710-rpi-3-b.dtb
enable_uart=1
kaba@kaba-Vostro-1550:~/Desktop/yocto/build/rpi3
$
```
In the host machine open serial console using `screen`. And connect the USB to ttl converter with PC. You should see the logs of Raspberry Pi booting.
```sh
$ sudo screen /det/ttyUSB0 115200
```

### Debug after boot complete
Configure `KDBoC` module to use `ttyS0` and enter `KDB` mode using `sysrq-trigger`. In `KDB` console, enter `kgdb` to make kernel listen to remote `GDB` debugger.
```sh
root@raspberrypi3-64:~# echo ttyS0 > /sys/module/kgdboc/parameters/kgdboc
[  219.105202] KGDB: Registered I/O driver kgdboc
root@raspberrypi3-64:~# echo g > /proc/sysrq-trigger
[  255.963036] sysrq: SysRq : DEBUG

Entering kdb (current=0xfffffff2f7f60000, pid 396) on processor 3 due to Keyboard Entry
[3]kdb> kgdb
Entering please attach debugger or use $D#44+ or $3#33


```
Raspberry Pi will be waiting for `GDB` client debugger connect serially.

### Debug during boot
If you want to debug something during boot, you have to connect `GDB` at very early stage of booting. Linux provides a command line argument option to achieve this. Configure `kgdboc` and `kgdbwait` in kernel `bootargs`. So kernel will wait after minimal initialization of hardware.

```sh
root@raspberrypi3-64:~# cat /boot/cmdline.txt
dwc_otg.lpm_enable=0 console=serial0,115200 kgdboc=ttyS0,115200 kgdbwait root=/dev/mmcblk0p2 rootfstype=ext4 rootwait
root@raspberrypi3-64:~# reboot
```
<div style="color:red;">
<span style="font-weight: bold;">WARNING:</span> As mentioned [here](https://github.com/raspberrypi/linux/issues/2245), it is a known issue that Raspberry Pi 3 doesn't boot with `kgdboc` set. So this will not work as of now. Let me find a work-around and update that in a future post.
</div>

### GDB connect
When Pi started waiting for `GDB` to connect, run the `cross-GDB` from host. You have to run this as a root.
```sh
root@kaba-Vostro-1550:/home/kaba/Desktop/yocto/build/rpi3/tmp/work/raspberrypi3_64-poky-linux/linux-raspberrypi/1_4.9.59+gitAUTOINC+e7976b2aff-r0/image/boot# /home/kaba/Desktop/yocto/build/rpi3/tmp/work/x86_64-nativesdk-pokysdk-linux/gdb-cross-canadian-aarch64/8.0-r0/image/opt/poky/2.4.2/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-gdb -tui ./vmlinux-4.9.59
```
Once `GDB` is connected to the board, it will look like this.
![Raspberry Pi GDB session](images/gdb_home_screenshot.png)

# References
 * [https://kaiwantech.wordpress.com/2013/07/04/a-kdb-kgdb-session-on-the-popular-raspberry-pi-embedded-linux-board/]
 * [https://github.com/FooDeas/raspberrypi-ua-netinst/issues/122]
 * [https://www.raspberrypi.org/forums/viewtopic.php?t=19186]
