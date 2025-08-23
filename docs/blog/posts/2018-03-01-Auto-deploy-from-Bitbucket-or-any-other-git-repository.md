---
title: Auto deploy from Bitbucket or any other git repository
date: 2018-03-01 19:47:18
tags:
 - bitbucket
 - nodejs
 - git
category: setup
thumbnail: "/images/git_hooks.jpeg"
---

Update aptitude first
```sh
$ sudo apt-get update
```

Install expect
```sh
$ sudo apt-get install expect
```

Create a directory named `trigger` in your home directory
```sh
$ cd
$ mkdir trigger
$ cd trigger
```

Create following files inside the `trigger` directory.
**trigger.js**
```sh
$ cat trigger.js
var server_port = <new_port_for_trigger>;

var sys = require('sys');
var exec = require('child_process').exec;
var child;

var http = require('http');
var express = require('express');
var app = express();
var server_get = require('http').Server(app);

app.post('/update', function(req, res) {
	child = exec("./update_repo.sh", function(error, stdout, stderr) {
				console.log('stdout: ' + stdout);
				console.log('stderr: ' + stderr);
				if (error !== null) {
					console.log('exec error: ' + error);
				}
	});

	res.send("SUCCESS");
});

app.listen(server_port, function() {
    console.log('Example app listening on port ' + server_port);
}
```

**package.json**
```sh
$ cat package.json
{
	"name": "Trigger",
	"version": "0.0.1",
	"scripts": {
		"start": "node server"
	},
	"dependencies": {
		"express": "^4.14.0",
		"http": "0.0.0",
		"https": "^1.0.0"
	}
}
```

**update_repo.sh**
```sh
$ cat update_repo.sh
#!/bin/bash
cd <your_git_repo>
~/trigger/git_pull_helper.sh
pm2 restart server
```

**git_pull_helper.sh**
```sh
$ cat git_pull_helper.sh
#!/usr/bin/expect -f
spawn git pull
expect "ass"
send "<your_ssh_key_pass_phrase>\r"
interact
```

Update permission for *update_repo.sh* and *git_pull_helper.sh*
```sh
$ chmod +x update_repo.sh
$ chmod +x git_pull_helper.sh
```

Start your trigger NodeJS server
```sh
$ pm2 start trigger.js
```

 * Configure you `nginx` if you are using one
 * Don't forget to enable firewall to allow the new port

### Update in bitbucket
 * Got *Settings*
 * Select *Webhooks* in *Workflow* section
 * Click *Add webhook* button
 * Give name
 * Enter your url as `*<your_domain_name>.<ext>:<your_port_for_trigger>/update*`
 * Press save button

Or follow the steps provided by your `git` server.
