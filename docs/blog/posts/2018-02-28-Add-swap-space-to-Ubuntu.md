---
title: Add swap space to Ubuntu
date: 2018-02-28 08:32:54
tags:
 - swap
 - digitalocean
 - memory management
category: setup
thumbnail: "/images/add_swap_to_ubuntu.jpg"
---

<aside class="alert alert-danger" style="border-radius: 3px;">
<strong>WARNING:</strong> Adding a swap space in your Digitalocean droplet or any other cloud VM is not recommended!
Because continuous reads and write is will reduce the life of underlying flash.
</aside><br/>

Truffle compilation was crashing with memory-overrun issues in my 512 MB basic droplet. So I added 2 GB of swap space with following commands.

Make sure you have no swap config previously. There should be no output.
```sh
$ sudo swapon --show
$
```

The simple way of doing this is by creating a swap file. Lets look at all the partitions to see how much free space in each of them.
```sh
$ df -h
kaba@ubuntu-512mb-blr1-01:~$ df -h
Filesystem      Size  Used Avail Use% Mounted on
udev            238M     0  238M   0% /dev
tmpfs            49M  5.9M   43M  13% /run
/dev/vda1        20G  5.2G   15G  27% /
tmpfs           245M   24M  221M  10% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
tmpfs           245M     0  245M   0% /sys/fs/cgroup
/dev/vda15      105M  3.4M  102M   4% /boot/efi
tmpfs            49M     0   49M   0% /run/user/0
tmpfs            49M  8.0K   49M   1% /run/user/1000
$
```
 * Other than the ones with `/dev/` are virtual filesystem and are not usable.
 * I'm gonna create my swap file in `/dev/vda1`.

Use `fallocate` utility to create a preallocated file in root directory.
```sh
$ sudo fallocate -l 2G /swapfile
```
 * As in previous command output, `dev/vda1` is mounted on `/`
 * 2G specifies 2 GB of space. Use whatever size you want in that place.

Make the is file accessible to `root` user alone
```sh
sudo chmod 600 /swapfile
```

Mark the file as swap space
```sh
$ sudo mkswap /swapfile
```

Enable the swap file, so your system will start using it
```sh
$ sudo swapon /swapfile
```

Done!

Now `swapon` command shouldn't return an empty result
```sh
$ sudo swapon --show
NAME      TYPE SIZE   USED PRIO
/swapfile file   2G 771.6M   -1
```

<aside class="alert alert-info" style="border-radius: 3px;">
<strong>Note:</strong> The swap file will be used only until the system reboots. You have to use the `swapon` command every time your system reboots. Or read further.
</aside>

### Make swap settings permanent
You need to edit `/etc/fstab` file. So better take a backup of it.
```sh
$ sudo mv /etc/fstab /etc/fstab.orig
```

Update the file with swap information
```sh
$ echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
