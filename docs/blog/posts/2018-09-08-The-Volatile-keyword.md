---
title: The Volatile keyword
date: 2018-09-08 21:38:50
thumbnail: "/images/the_volatile_keyword.jpeg"
description: "A detailed discussion about volatile keyword."
tags:
 - c
 - gcc
 - compiler
---

Recently I've interviewed some candidates for entry and intermediate level positions. One of the questions most of them struggled is about the `volatile` keyword. Some conversations went like this,
 * Q: Why we use `volatile` keyword?
 * A: It will tell compiler not to use any registers for the `volatile` variable.
 * Q: Then how will it work in an ARM processor? In ARM no instruction other than load and store can use memory location.
 * A: ??!!


 * Q: What is the purpose `volatile` keyword?
 * A: We'll use for IO memory
 * Q: Why we need it for IO memory?
 * A: So every time processor accesses the memory, it will go to IO device
 * Q: So `volatile` is to tell processor not to cache data?
 * A: Yes
 * Q: Thus `volatile` is a processor directive not compiler directive?
 * And confusion starts


In this post, lets see how `volatile` works with two simple C programs. In complex programs with multiple variables and loops `volatile` keyword will make significant difference in speed and memory usage.

GCC provides many compiler optimization flags. Enabling them will aggressively optimize the code and give better performance in terms of speed and memory footprint. As these optimizations make debugging harder, they are not suitable development. All available GCC compiler optimization flags can be get from following command.
```sh
$ $CC --help=optimizers
The following options control optimizations:
  -O<number>                  Set optimization level to <number>.
  -Ofast                      Optimize for speed disregarding exact standards compliance.
  -Og                         Optimize for debugging experience rather than speed or size.
  -Os                         Optimize for space rather than speed.
  -faggressive-loop-optimizations Aggressively optimize loops using language constraints.
.
.
.
```

For simplicity, I used only `-Ofast` optimizer for the examples. It informs GCC to do its best to make the program run faster. We'll see how compiler builds, with and without `volatile`. GCC will give assembly code output with `-S` options.

Take the following C program.
```C
#include <stdio.h>

int main() {
	int *x = (int *)0xc000;
	int b = *x;
	int c = *x;
	*x = b + c;
	return 0;
}
```
Don't worry about dereferencing a random virtual address. We are not going to run this program, just build the assembly code and examine manually. I use pointers in these programs. Because using immediate value makes no sense with volatile. We have an integer pointer `x` points to address `0xc000`. We initialize two variables `b` and `c` with value in address `0xc000`. And then addition of `b` and `c` is stored in location `0xc000`. So we read the value in location `0xc000` twice in this program. Let see how it gets compiled by GCC form ARMv8.

```sh
$ echo $CC
aarch64-poky-linux-gcc --sysroot=/opt/poky/2.4.2/sysroots/aarch64-poky-linux
kaba@kaba-Vostro-1550:~/Desktop/volatile/single_varuable_two_reads
$ $CC -S -Ofast ./code.c -o code.S
kaba@kaba-Vostro-1550:~/Desktop/volatile/single_varuable_two_reads
$
```

```ASM
	.arch armv8-a
	.file	"code.c"
	.text
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	main
	.type	main, %function
main:
	mov	x2, 49152
	mov	w0, 0
	ldr	w1, [x2]
	lsl	w1, w1, 1
	str	w1, [x2]
	ret
	.size	main, .-main
	.ident	"GCC: (GNU) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```
The compiler intelligently finds that variable `b` and `c` have same value from address `0xc000` and addition of them is equivalent to multiplying the value at `0xc000` by two. So it loads the value into register `W1` and left shifts it by 1 (equivalent of multiplying with two) and then stores the new value into location `0xc000`.

Now lets change the code to use `volatile` for variable `x`. And see how the assembly code looks.
```C
#include <stdio.h>

int main() {
	volatile int *x = (int *)0xc000;
	int b = *x;
	int c = *x;
	*x = b + c;
	return 0;
}
```
```ASM
	.arch armv8-a
	.file	"code.c"
	.text
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	main
	.type	main, %function
main:
	mov	x1, 49152
	mov	w0, 0
	ldr	w2, [x1]
	ldr	w3, [x1]
	add	w2, w2, w3
	str	w2, [x1]
	ret
	.size	main, .-main
	.ident	"GCC: (GNU) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```
This time the compiler considers that the value at location `0xc000` may be different each time it reads. It thinks that the variables `b` and `c` could be initialized with different values. So it reads the location `0xc000` twice and adds both values.

Lets see a simple loop case
```C
#include <stdio.h>

int main() {
	int *x = (int *)0xc000;
	int *y = (int *)0xd000;
	int sum = 0;
	for (int i = 0; i < *y; i++) {
		sum = sum + *x;
	}
	*x = sum;
	return 0;
}
```
This program initializes two pointers `x` and `y` to locations `0xc000` and `0xd000` respectively. It adds the value at `x` to itself as many times the value at `y`. Lets see how GCC sees it.
```ASM
	.arch armv8-a
	.file	"code.c"
	.text
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	main
	.type	main, %function
main:
	mov	x0, 53248
	ldr	w0, [x0]
	cmp	w0, 0
	ble	.L3
	mov	x1, 49152
	ldr	w1, [x1]
	mul	w1, w0, w1
.L2:
	mov	x2, 49152
	mov	w0, 0
	str	w1, [x2]
	ret
.L3:
	mov	w1, 0
	b	.L2
	.size	main, .-main
	.ident	"GCC: (GNU) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```
The compiler assigns register `X0` to `y` and register `X1` to `x`. The program compares the value at `[X0]` - value at the address in `X0` - with zero. If so, it jumps to `.L3` which sets `W1` to zero and jumps to `.L2`. Or it simply multiplies `[X0]` and `[X1]` and stores the value in `W1`. `.L2` stores the value in `W1` at `[X2]` and returns. The compiler intelligently identifies that adding `[X2]` to itself `[X1]` times is equivalent to multiplying both.

With `volatile`,
```C
#include <stdio.h>

int main() {
	volatile int *x = (int *)0xc000;
	int *y = (int *)0xd000;
	int sum = 0;
	for (int i = 0; i < *y; i++) {
		sum = sum + *x;
	}
	*x = sum;
	return 0;
}
```
the corresponding assembly code is
```ASM
	.arch armv8-a
	.file	"code.c"
	.text
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	main
	.type	main, %function
main:
	mov	x0, 53248
	ldr	w3, [x0]
	cmp	w3, 0
	ble	.L4
	mov	w0, 0
	mov	w1, 0
	mov	x4, 49152
	.p2align 2
.L3:
	ldr	w2, [x4]
	add	w0, w0, 1
	cmp	w0, w3
	add	w1, w1, w2
	bne	.L3
.L2:
	mov	x2, 49152
	mov	w0, 0
	str	w1, [x2]
	ret
.L4:
	mov	w1, 0
	b	.L2
	.size	main, .-main
	.ident	"GCC: (GNU) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```
This time GCC uses `X4` for the address `0xc000`, but its not significant for our problem. Look here the loop is `.L3`. It loads the value at location `X4` every time the loop runs, which is different than non-volatile behaviour. This time the compiler things the value at `X4` will be different each time it is read. So without any assumption, it adds the value to `sum` every time the loop runs.

In both programs the value at the location `0xc000` can be cached by the processor. The subsequent read of the value at `0xc000` could be from processor's cache but not from main memory. It is responsibility of the memory controller to maintain coherency between memory and processor cache. The `volatile` keyword has nothing to do here.

I believe these simple programs had explained the concept clear. The `volatile`
#### IS
 * To tell compiler not to make any assumption about the value stored in the variable

#### IS NOT
 * To tell the compiler not to use any registers to hold the value
 * To tell the processor not to cache the value
