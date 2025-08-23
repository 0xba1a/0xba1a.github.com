---
title: Variadic functions with unknown argument count
date: 2019-06-29 18:59:00
thumbnail: "/images/Variadic-functions-with-unknown-argument-count.jpg"
description: "Variadic function is function that accepts varying number arguments. But often times we've to pass the number of arguments as the first
argument. But once we faced a peculiar problem where the number of arguments can't be passed. And we solved it in a peculiar ;) way."
tags:
 - programming
 - C
---

One of my colleagues came across a peculiar problem. She had to write an API that accepts variable number of arguments, but number of arguments won't be passed in the arguments list. She cracked it intelligently with following hack.

## The Hack
Heart of this hack is a macro that can count the number of arguments passed to it. It has a limitation. Maximum number of arguments can be passed to this macro should be known. For example, if maximum number of arguments can be passed is 5, the macro will look like,

```sh
#define COUNT5(...) _COUNT5(__VA_ARGS__, 5, 4, 3, 2, 1)
#define _COUNT5(a, b, c, d, e, count, ...) count
```

If you want your macro to count 10 or lesser arguments,

```sh
#define COUNT10(...) _COUNT10(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _COUNT10(a, b, c, d, e, f, g, h, i, j, count, ...) count
```

Let me explain it. Consider below macro call. It will expand like this.

```sh
COUNT5(99, 98, 97);
  |
  |
  V
_COUNT5(99, 98, 97, 5, 4, 3, 2, 1)
  |
  |
  V
  3
```

The three arguments passed to `COUNT5` will occupy `a`, `b`, `c` of `_COUNT5`. `5` and `4` will occupy `d`, `e`. Next argument `3` will be in the place of `count`, that will be returned.

## Final solution
So she exposed a macro that accepts variable number of arguments as the API requested. This macro internally used the `COUNTX` macro to get number of arguments passed. And she passed the `count` and variable arguments to the actual `C` function. 

## Example
A small `C` program using this hack.

```C
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

int _sum(int count, ...);

#define COUNT(...) _COUNT(__VA_ARGS__, 5, 4, 3, 2, 1)
#define _COUNT(a, b, c, d, e, count, ...) count

#define sum(...) _sum(COUNT(__VA_ARGS__), __VA_ARGS__)

int _sum(int count, ...) {
	va_list	arg_ptr;
	int		sum = 0;
	int		i = 0;

	va_start(arg_ptr, count);

	for (i = 0; i < count; i++) {
		sum += va_arg(arg_ptr, int);
	}

	return sum;
}

int main() {
	printf("%d\n", sum(1, 2, 3, 4, 5));
	printf("%d\n", sum(1, 2, 3));
	printf("%d\n", sum(1));
	printf("%d\n", sum(2, 2, 2, 2, 2));

	return 0;
}
```

And its output.

```sh
kaba@kaba-Vostro-1550:~/variadic
$ gcc variadic.c
kaba@kaba-Vostro-1550:~/variadic
$ ./a.out
15
6
1
10
kaba@kaba-Vostro-1550:~/variadic
$
```
