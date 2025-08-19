---
title: VNC setup in Digitalocean
date: 2018-03-01 21:26:06
thumbnail: "/images/vnc_in_digitalocean.png"
tags:
 - vnc
 - digitalocean
 - ubuntu
 - setup
---

Install `xfce` desktop environment
```sh
$ sudo apt-get install xfce4 xfce4-goodies
```

Install `TightVNC` for VNC server
```sh
$ sudo apt-get install tightvncserver
```

Initialize VNC server
+ run VNC server once
+ enter and verify password
+ ignore view-only password
```sh
$ vncserver
Password:
Verify:
View-only password:
```

To configure VNC, first kill the running VNC server
```sh
$ vncserver -kill :1
Killing Xtightvnc process ID 13456
```

Backup `xstartup` configuration file
```sh
$ mv ~/.vnc/xstartup ~/.vnc/xstartup.orig
```

Edit `xstartup` file
```sh
$ vim ~/.vnc/xstartup
```

And add following content
```sh
*~/.vnc/xstartup*
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &
```
+ *Look at `~/.Xresources` file for VNC's GUI framework information*
+ *Start XFCE whenever VNC server is started*

Provide executable permission for `~/.vnc/xstartup` file
```sh
$ sudo chmod +x ~/.vnc/xstartup
```

Start VNC server
```sh
$ vncserver -geometry 1280x800
```
