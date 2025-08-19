---
title: Copy paste in tmux session inside ssh
date: 2021-07-31 12:05:28
thumbnail: "/images/tmux-copy-fix.png"
description: "A simple hack for tmux copy-paste to work in all environments. It can be RDB, ssh, ssh in tmux, tmux in ssh, etc.,"
tags:
 - tmux
 - linux
 - hack
---

When you access your tmux session on a remote machine from different client machines, based on the client machine configuration and terminal, etc., some features will not work. One of them is copy-paste.

I used to have tmux-yank configured to `xclip`. But it didn't work will when I accessed my remote VM from a Windows machine running putty. Similarly I had to configure every new machine I start using which was time consuming. Majority of my copy-paste activities will be between tmux panes and windows. I don't usually copy from the local machine to the remote machine. So I used this simple hack.

Configured my `.tmux.conf` to redirect `yank`-ed to a temporary file as below.
```shell
bind -T copy-mode-vi y send-keys -X copy-pipe-and-cancel 'cat > /tmp/clipboard'
```

Configured `.zshrc` to load an environment variable from that temporary file on every prompt.
```shell
precmd() { export p=`cat /tmp/clipboard` }
```

So if I copy any text in tmux using `y`, it will be populated into a environmental variable named `p`.

**NOTE:** *As the environmental variable will be loaded upon next prompt only, after copying, you need to press `enter` once it to take effect*

Below gif will show how it works
![tmux copy paste hack](/images/tmux_copy_paste_demo.gif)
