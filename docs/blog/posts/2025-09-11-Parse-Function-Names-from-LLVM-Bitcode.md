---
title: Parse Function Names from LLVM Bitcode
description: Simple introduction to LLVM and how to use it to parse all the function names in your code.
date: 2025-09-11
tags:
    - LLVM
    - Compilers
    - Context Engineering
    - clang
---
Large Language Models (LLMs) are increasingly used for coding tasks, but handling extensive codebases effectively requires special techniques. Simply feeding the entire codebase to an LLM can lead to hallucinations due to excessive context and significantly increase costs without yielding meaningful results. We'll discuss extensively about this in coming posts. This post focuses on a simple technique.

One effective technique is to avoid providing the entire code file, which often spans thousands of lines. Instead, extract and share only the skeleton of the file—retaining global variables, classes, and their member functions. This approach minimizes context while preserving essential information. You can explore this concept further in the [research paper](https://arxiv.org/pdf/2409.16299).

But how can we achieve this? How do we generate a skeleton of a source file—essentially a list of exported symbols along with their hierarchy? The paper mentioned above suggests using `ctags`, a simple reference lookup tool. However, `ctags` lacks the ability to provide hierarchy information. A better alternative lies in leveraging compilers. In this blog post, we’ll explore how to use LLVM to list all the functions in a C program.

**Note:** *As this is the very first post on LLVM, basic detail about the LLVM command-line parser and how to compile an LLVM program using a Makefile will be explained here. This post will act as an onboarding document supporting subsequent LLVM related posts.*

## Setup

To follow along, you need to have `llvm` and `clang` installed in your system.

``` shell linenums="1" title="Install llvm and clang"
 $ sudo apt install llvm clang -y
```

## Test Program

Let's take a simple program that does arithmetic operations.

``` c linenums="1" title="test.c"
--8<-- "code/llvm/parse_function_names/test.c"
```

Here we have three functions, `main`, `add` and `sub`. Our parser will list these functions along with some additional information.

## The LLVM Parser

Complete parser code can be found at the bottom of the post. Here, let me explain the important parts in detail. We initially have the header file inclusions for all the different LLVM modules. Then we use the `llvm` namespace for easier typing.

Then we declare a static variable `Filename` to gather the argument passed.

``` cpp linenums="1" title="parse_function_names.cpp"
static cl::opt<std::string> FileName(cl::Positional, cl::desc("Bitcode file"), cl::Required);

int main(int argc, char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "LLVM hello world\n");
```

Though it is not directly related to parsing or compiling code, it is a feature provided by LLVM under the namespace `llvm:cl`. Using this we declare a static variable called `FileName` and mark it as a required positional argument. Then inside the main when we call `cl::ParseCommandLineOptions` function, the static variable will be automatically filled by the user provided argument. Pretty handy right?

Then, load the input file into memory.

``` cpp linenums="1" title="parse_function_names.cpp"
    LLVMContext context;
    std::string error;
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr = MemoryBuffer::getFile(FileName);

    if (!BufferOrErr)
    {
        std::cerr << "Error reading file: " << BufferOrErr.getError().message() << "\n";
        return 1;
    }

    std::unique_ptr<MemoryBuffer> &mb = BufferOrErr.get();
```

Lexing and parsing are resource-intensive tasks. Reading code from disk one token at a time can severely impact performance. To address this, we use a memory buffer to load the entire file into memory. Most interfaces, such as `MemoryBuffer::getFile`, include built-in error handling. Always check the return value for errors before proceeding. This defensive coding style is essential when working with LLVM APIs.

**Please remember that `test.c` will not be passed as an argument. We'll compile the it and the compiled binary file will be passed as an argument to this parser.** So, the memory buffer contains the llvm bitcode no the C code.

Now parse the functions from the loaded bitcode file.

``` cpp linenums="1" title="parse_function_names.cpp"
    auto ModuleOrErr = parseBitcodeFile(mb->getMemBufferRef(), context);
    if (!ModuleOrErr)
    {
        std::cerr << "Error parsing bitcode: " << toString(ModuleOrErr.takeError()) << "\n";
        return 1;
    }

    std::unique_ptr<Module> m = std::move(ModuleOrErr.get());

    raw_os_ostream O(std::cout);
    for (Module::const_iterator i = m->getFunctionList().begin(),
                                e = m->getFunctionList().end();
         i != e; ++i)
    {
        if (!i->isDeclaration())
        {
            O << i->getName() << " has " << i->size() << " basic block(s).\n";
        }
    }
```

In LLVM term, the functions are usually called as modules. So, we gather all the modules from the bitcode. Then we iterate one by one and print only the function definitions. Very simple and straight-forward.


## Parser In Action

### Makefile

We need to compile the original source to be parsed first. Then the LLVM parser should be compiled. Then pass the bitcode compilation of the original source to the LLVM parser. Compiling the LLVM parser by hand is not recommended due to the lengthy flags to be passed. So, let's write a Makefile. The complete file can be seen at the bottom of the post. Here I'll explain the important snippets.

``` basemake linenums="1" title="Makefile"
LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif
```

`llvm-config` is a very helpful tool. It provides standardized and portable way to get the necessary compiler and linker flags to make the developers' life easier. Then we have an option to be verbose or quieter.

With the help of `llvm-config` we construct our compiler flags.

``` basemake linenums="1" title="Makefile"
SRC_DIR?=$(PWD)
LDFLAGS+=$($(shell $(LLVM_CONFIG) --ldflags))
COMMON_FLAGS=-Wall -Wextra
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)
```

Below are typical multi-level Makefile with targets - source(`.cpp`) --> object(`.o`) --> binary - along with a `clean`.

``` basemake linenums="1" title="Makefile"
PARSE_FUNCTION_NAMES=parse_function_names
PARSE_FUNCTION_NAMES_OBJECTS=parse_function_names.o

%.o : $(SRC_DIR)/%.cpp
	@echo Compiling $*.cpp
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(PARSE_FUNCTION_NAMES) : $(PARSE_FUNCTION_NAMES_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ `$(LLVM_CONFIG) --libs bitreader core support`

clean:
	@echo Cleaning up...
	$(QUIET)rm -f $(PARSE_FUNCTION_NAMES) $(PARSE_FUNCTION_NAMES_OBJECTS)

.PHONY: all clean
```

If you notice keenly, we didn't override the `CXX`. So, it falls backs to the default `g++`. It is okay. We don't need to compile our parser using llvm. We just need to compile and link it along with llvm libraries. So, we compile our parser with `gcc` itself.

### Execution

First, compile the input source into llvm bitcode. We installed `clang` for this purpose only. So far we never used any `clang` libraries in the parser. `clang` is the preferred LLVM front-end for C, CPP and Objective-C programs. So, we will use it to convert our `test.c` into llvm bitcode.

**Note:** *LLVM employs a front-end --> IR --> back-end compilation model. This means that LLVM uses an Intermediate Representation (IR) as an object representation. During compilation, source code is first converted to IR by a front-end. The IR is then transformed into machine code by a back-end. This design allows developers to implement a front-end without needing to account for the target machine, enabling the same front-end to work with multiple architectures by pairing it with the appropriate back-end.*

``` bash linenums="1" title="Compile the test code"
 $ clang -c -emit-llvm test.c -o test.bc
 $ file test.bc
test.bc: LLVM IR bitcode
 $
```

Now compile our parser and pass the `test.bc` to it.

``` bash linenums="1" title="Build and run the parser"
 $ make
 ...
 ...
 Linking parse_function_names
 $ echo $?
 0
 $ $ ./parse_function_names test.bc
add has 1 basic block(s).
sub has 1 basic block(s).
main has 1 basic block(s).
$
```

Congratulations! We have successfully extracted and listed the functions defined in a source file by parsing its LLVM bitcode. Complete code of `parse_function_names.cpp` and the `Makefile` can be found below. Also, I've explained an alternate method do the same task at the end of this article.

## Complete Code

``` cpp linenums="1" title="parse_function_names.cpp"
--8<-- "code/llvm/parse_function_names/parse_function_names.cpp"
```

``` basemake linenums="1" title="Makefile"
--8<-- "code/llvm/parse_function_names/Makefile"
```

## Alternative way

There is a tool called `llvm-dis` which will convert the llvm IR into a human readable format. When the bitcode is in human readable format, we can simply list the function definitions using the keyword `define`.

``` bash linenums="1"
 $ llvm-dis test.bc -o test.ll
 $ grep "define" test.ll
define dso_local i32 @add(i32 noundef %0, i32 noundef %1) #0 {
define dso_local i32 @sub(i32 noundef %0, i32 noundef %1) #0 {
define dso_local i32 @main() #0 {
```

While there are plenty of tools out there to automate parsing, rolling your own parser has real advantages. You get to tailor the logic to your exact needs—something off-the-shelf solutions often can't do. This flexibility is especially useful when you're building systems like IDEs designed for LLM workflows, where generic tools just don't cut it. Plus, understanding the nuts and bolts of parsing gives you more control and insight into how your tooling works.
