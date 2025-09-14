LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif

SRC_DIR?=$(PWD)
LDFLAGS+=$($(shell $(LLVM_CONFIG) --ldflags))
COMMON_FLAGS=-Wall -Wextra
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

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