---
title: Virtual memory to Physical memory
date: 2018-03-02 20:14:24
thumbnail: "/images/virtual_memory_to_physical_memory.jpg"
tags:
 - proc
 - memory management
 - virtual memory
---

We all know that processes running in Linux acts only in virtual address space. So whenever a process wants to access a data *(okay datum)* it requests CPU for a virtual address. The CPU intern converts it into physical address and fetches the data. It will be nice to have a program that converts virtual address to physical address, won't it?

Linux from 2.5.26 provides a `proc` interface, `pagemap` that contains information what we want. Each process has its `pagemap` at `/proc/p_id/pagemap`. According to the [Documentation](https://www.kernel.org/doc/Documentation/vm/pagemap.txt) it is a binary file contains a sequence of 64-bit words. Each word contains information regarding one virtual page for full virtual address space. Among them bits 0-54 (55-bits) represents the address of the physical frame number (`PFN`). I think that's all we need. Adding the `offset` of a variable from virtual page address to the `PFN` will give us the physical memory address.

**WARNING:** Don't try to read the `pagemap` file directly. `cat /proc/self/pagemap` or `vim /proc/p_id/pagemap` is not going to return anytime soon.

We'll write a small C program and the let's try to get physical address of a variable used in that C program. As the `PFN` data will be present only if the data is not moved to *swap*, lets use `mlock()` to lock the memory in physical memory.

``` C
#include <stdio.h>
#include <sys/mman.h> 	/* for mlock() */
#include <stdlib.h>		/* for malloc() */
#include <string.h>		/* for memset() */

/* for getpid() */
#include <sys/types.h>
#include <unistd.h>

#define MEM_LENGTH 1024

int main()
{
	/* Allocate 1024 bytes in heap */
	char *ptr = NULL;
	ptr = malloc(MEM_LENGTH);
	if (!ptr) {
		perror("malloc fails. ");
		return -1;
	}

	/* obtain physical memory */
	memset(ptr, 1, MEM_LENGTH);

	/* lock the allocated memory in RAM */
	mlock(ptr, MEM_LENGTH);

	/* print the pid and vaddr. Thus we can work on him */
	printf("my pid: %d\n\n", getpid());
	printf("virtual address to work: 0x%lx\n", (unsigned long)ptr);

	/* make the program to wait for user input */
	scanf("%c", &ptr[16]);

	return 0;
}
```
<br>


Run the `specimen.c` program, get its `p_id` and start the dissection.

``` sh
$ gcc specimen.c -o specimen
$ ./specimen
my pid: 11953

virtual address to work: 0x55cd75821260


```
<br>


In a 64-bit machine, virtual address-space is from *0x00* and to *2^64 - 1*. First we have to calculate the page offset for the given virtual address [find on which virtual memory page, the address resides]. And multiply that with 8 as each virtual page table has 8-byte information word in the `pagemap` file.

``` C
#define PAGEMAP_LENGTH 8
page_size = getpagesize();
offset = (vaddr page_size) * PAGEMAP_LENGTH;
```
<br>


Open the `pagemap` file and seek to that offset location.

``` C
pagemap = fopen(filename, "rb");
fseek(pagemap, (unsigned long)offset, SEEK_SET)
```
<br>


Now cursor is on the first byte of 64-bit word containing the information we need. According to the [Documentation](https://www.kernel.org/doc/Documentation/vm/pagemap.txt) bits 0-54 represents the physical page frame number (`PFN`). So read 7-bytes and discard most significant bit.

``` C
fread(&paddr, 1, (PAGEMAP_LENGTH-1), pagemap)
paddr = paddr & 0x7fffffffffffff;
```
<br>


This is the `PFN`. Add offset of the virtual address from its virtual page base address to the page shifted `PFN` to get the physical address of the memory.

``` C
offset = vaddr % page_size;
/* PAGE_SIZE = 1U << PAGE_SHIFT */
while (!((1UL << ++page_shift) & page_size));
paddr = (unsigned long)((unsigned long)paddr << page_shift) + offset;
```
<br>


Here is the complete program.

``` C
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define PAGE_SHIFT 12
#define PAGEMAP_LENGTH 8

int main(int argc, char **argv)
{
	unsigned long vaddr, pid, paddr = 0, offset;
	char *endptr;
	FILE *pagemap;
	char filename[1024] = {0};
	int ret = -1;
	int page_size, page_shift = -1;

	page_size = getpagesize();
	pid = strtol(argv[1], &endptr, 10);
	vaddr = strtol(argv[2], &endptr, 16);
	printf("getting page number of virtual address %lu of process %ld\n",vaddr, pid);

	sprintf(filename, "/proc/%ld/pagemap", pid);

	printf("opening pagemap %s\n", filename);
	pagemap = fopen(filename, "rb");
	if (!pagemap) {
		perror("can't open file. ");
		goto err;
	}

	offset = (vaddr / page_size) * PAGEMAP_LENGTH;
	printf("moving to %ld\n", offset);
	if (fseek(pagemap, (unsigned long)offset, SEEK_SET) != 0) {
		perror("fseek failed. ");
		goto err;
	}

	if (fread(&paddr, 1, (PAGEMAP_LENGTH-1), pagemap) < (PAGEMAP_LENGTH-1)) {
		perror("fread fails. ");
		goto err;
	}
	paddr = paddr & 0x7fffffffffffff;
	printf("physical frame address is 0x%lx\n", paddr);

	offset = vaddr % page_size;

	/* PAGE_SIZE = 1U << PAGE_SHIFT */
	while (!((1UL << ++page_shift) & page_size));

	paddr = (unsigned long)((unsigned long)paddr << page_shift) + offset;
	printf("physical address is 0x%lx\n", paddr);

	ret = 0;
err:
	fclose(pagemap);
	return ret;
}
```
<br>


And the output

``` sh
$ sudo ./a.out 11953 0x55cd75821260
getting page number of virtual address 94340928115296 of process 11953
opening pagemap /proc/11953/pagemap
moving to 184259625224
physical frame address is 0x20508
physical address is 0x20508260
```
<br>

### References
 1. https://www.kernel.org/doc/Documentation/vm/pagemap.txt
 2. https://elixir.bootlin.com/linux/latest/source/arch/x86/include/asm/page_types.h
 3. man pages
