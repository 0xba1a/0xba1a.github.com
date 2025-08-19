---
title: Configure Vim
date: 2017-09-01 19:59:40
tags:
 - setup
 - vim
thumbnail: "images/configure_vim.jpg"
---

The above picture is from my favorite serial `Mr.Robot`. Elliot Alderson is writing a script to hack Evil Corp.

I have three development environments
 * One Cent-OS machine for C programming in office
 * Ubuntu-16.04 Digitalocean and AWS virtual machines to host some servers and Ethereum developemnt
 * And one Raspberry PI 3 at home as an SSH client

As the Cent-OS machine has older version of vim, Plugins that require latest version of vim and plugins that are only needed for Web/Ethereum development are kept under an <em>if</em> condition. So modify it based on your need.

Here I didn't give any explanation for any config or plugin. Copy pasting them in Google will give you the information.

Backup you `vimrc` file
```bash
$ mv ~/.vimrc ~/.vimrc.orig
```

Clone Vundle into your `~/.vim` directory
```sh
$ git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

Copy this `vimrc` file into your home directory.
<script type="text/javascript" src="https://gist-it.appspot.com/http://github.com/kaba-official/vimrc/raw/master/.vimrc">
</script>

<aside class="alert alert-info">
I use `h, j, k, l` for navigation. Modify the `vimrc` file if necessary.
</aside>

Install plugins using `PluginInstall` command. Open vim with no file. Ignore if any error shows up. Press `Esc` key then `:` and type the command `PluginInstall` followed by `Enter` key. A new window will open and install all plugins.
```bash
$ vim

:PluginInstall
```

Don't forget to install 'YouCompleteMe' in your machine.
```bash
$ cd ~/.vim/bundle/YouCompleteMe/
$ ./install.py --js-completer --clang-completer --system-libclang
```
Installing auto-complete feature for JavaScript and C Language. See [YouCompleteMe GitHub page](https://github.com/Valloric/YouCompleteMe) for more details.
