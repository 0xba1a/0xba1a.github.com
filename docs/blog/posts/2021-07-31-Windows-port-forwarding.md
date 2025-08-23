---
title: Windows-10 port forwarding
description: "Just a command to enable port forwarding in Windows"
thumbnail: "/images/windows-rdb.png"
date: 2021-07-31 13:04:38
tags:
 - windows
 - hack
 - portproxy
---

Below is the command to port-forward in Windows-10. It can be executed in normal command-prompt not in PowerShell. Make sure you've administrator privilege. This will be useful in when you run a VM in your Windows workstation. So you can directly ssh to the VM instead of opening RDB every time.

```shell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=18001 connectaddress=192.168.105.115 connectport=22
```

Now this machine forwards any incoming connection on port number *18001* to *192.168.105.115* IP's port *22*. To ssh into the machine *192.168.105.115* from an external network, you should issue the following command (provided that windows machine IP is *112.172.29.29*).

```shell
 $ ssh 112.172.29.29 -p 18001
```
