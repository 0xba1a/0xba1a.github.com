#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include <iostream>

using namespace llvm;

static cl::opt<std::string> FileName(cl::Positional, cl::desc("Bitcode file"), cl::Required);

int main(int argc, char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "LLVM hello world\n");
    LLVMContext context;
    std::string error;
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr = MemoryBuffer::getFile(FileName);

    if (!BufferOrErr)
    {
        std::cerr << "Error reading file: " << BufferOrErr.getError().message() << "\n";
        return 1;
    }

    std::unique_ptr<MemoryBuffer> &mb = BufferOrErr.get();

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
    return 0;
}