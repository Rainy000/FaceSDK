# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wangy/3rdParty/ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wangy/3rdParty/ncnn/build_linux

# Include any dependencies generated for this target.
include examples/CMakeFiles/rfcn.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/rfcn.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/rfcn.dir/flags.make

examples/CMakeFiles/rfcn.dir/rfcn.cpp.o: examples/CMakeFiles/rfcn.dir/flags.make
examples/CMakeFiles/rfcn.dir/rfcn.cpp.o: ../examples/rfcn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangy/3rdParty/ncnn/build_linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/rfcn.dir/rfcn.cpp.o"
	cd /home/wangy/3rdParty/ncnn/build_linux/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rfcn.dir/rfcn.cpp.o -c /home/wangy/3rdParty/ncnn/examples/rfcn.cpp

examples/CMakeFiles/rfcn.dir/rfcn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rfcn.dir/rfcn.cpp.i"
	cd /home/wangy/3rdParty/ncnn/build_linux/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangy/3rdParty/ncnn/examples/rfcn.cpp > CMakeFiles/rfcn.dir/rfcn.cpp.i

examples/CMakeFiles/rfcn.dir/rfcn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rfcn.dir/rfcn.cpp.s"
	cd /home/wangy/3rdParty/ncnn/build_linux/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangy/3rdParty/ncnn/examples/rfcn.cpp -o CMakeFiles/rfcn.dir/rfcn.cpp.s

examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.requires:

.PHONY : examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.requires

examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.provides: examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/rfcn.dir/build.make examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.provides.build
.PHONY : examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.provides

examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.provides.build: examples/CMakeFiles/rfcn.dir/rfcn.cpp.o


# Object files for target rfcn
rfcn_OBJECTS = \
"CMakeFiles/rfcn.dir/rfcn.cpp.o"

# External object files for target rfcn
rfcn_EXTERNAL_OBJECTS =

examples/rfcn: examples/CMakeFiles/rfcn.dir/rfcn.cpp.o
examples/rfcn: examples/CMakeFiles/rfcn.dir/build.make
examples/rfcn: src/libncnn.a
examples/rfcn: /home/wangy/3rdParty/opencv348_sdk/lib/libopencv_highgui.so.3.4.8
examples/rfcn: /home/wangy/3rdParty/opencv348_sdk/lib/libopencv_videoio.so.3.4.8
examples/rfcn: /home/wangy/3rdParty/opencv348_sdk/lib/libopencv_imgcodecs.so.3.4.8
examples/rfcn: /home/wangy/3rdParty/opencv348_sdk/lib/libopencv_imgproc.so.3.4.8
examples/rfcn: /home/wangy/3rdParty/opencv348_sdk/lib/libopencv_core.so.3.4.8
examples/rfcn: examples/CMakeFiles/rfcn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangy/3rdParty/ncnn/build_linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rfcn"
	cd /home/wangy/3rdParty/ncnn/build_linux/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rfcn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/rfcn.dir/build: examples/rfcn

.PHONY : examples/CMakeFiles/rfcn.dir/build

examples/CMakeFiles/rfcn.dir/requires: examples/CMakeFiles/rfcn.dir/rfcn.cpp.o.requires

.PHONY : examples/CMakeFiles/rfcn.dir/requires

examples/CMakeFiles/rfcn.dir/clean:
	cd /home/wangy/3rdParty/ncnn/build_linux/examples && $(CMAKE_COMMAND) -P CMakeFiles/rfcn.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/rfcn.dir/clean

examples/CMakeFiles/rfcn.dir/depend:
	cd /home/wangy/3rdParty/ncnn/build_linux && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangy/3rdParty/ncnn /home/wangy/3rdParty/ncnn/examples /home/wangy/3rdParty/ncnn/build_linux /home/wangy/3rdParty/ncnn/build_linux/examples /home/wangy/3rdParty/ncnn/build_linux/examples/CMakeFiles/rfcn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/rfcn.dir/depend

