# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/jaime/Desktop/percepcion_tmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jaime/Desktop/percepcion_tmp/build

# Include any dependencies generated for this target.
include CMakeFiles/PCL.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PCL.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PCL.dir/flags.make

CMakeFiles/PCL.dir/main.cpp.o: CMakeFiles/PCL.dir/flags.make
CMakeFiles/PCL.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaime/Desktop/percepcion_tmp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PCL.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PCL.dir/main.cpp.o -c /home/jaime/Desktop/percepcion_tmp/main.cpp

CMakeFiles/PCL.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCL.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaime/Desktop/percepcion_tmp/main.cpp > CMakeFiles/PCL.dir/main.cpp.i

CMakeFiles/PCL.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCL.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaime/Desktop/percepcion_tmp/main.cpp -o CMakeFiles/PCL.dir/main.cpp.s

# Object files for target PCL
PCL_OBJECTS = \
"CMakeFiles/PCL.dir/main.cpp.o"

# External object files for target PCL
PCL_EXTERNAL_OBJECTS =

PCL: CMakeFiles/PCL.dir/main.cpp.o
PCL: CMakeFiles/PCL.dir/build.make
PCL: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_people.so
PCL: /usr/lib/x86_64-linux-gnu/libboost_system.so
PCL: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
PCL: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
PCL: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
PCL: /usr/lib/x86_64-linux-gnu/libboost_regex.so
PCL: /usr/lib/x86_64-linux-gnu/libqhull.so
PCL: /usr/lib/libOpenNI.so
PCL: /usr/lib/libOpenNI2.so
PCL: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libfreetype.so
PCL: /usr/lib/x86_64-linux-gnu/libz.so
PCL: /usr/lib/x86_64-linux-gnu/libjpeg.so
PCL: /usr/lib/x86_64-linux-gnu/libpng.so
PCL: /usr/lib/x86_64-linux-gnu/libtiff.so
PCL: /usr/lib/x86_64-linux-gnu/libexpat.so
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_features.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_search.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_io.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
PCL: /usr/lib/x86_64-linux-gnu/libpcl_common.so
PCL: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libfreetype.so
PCL: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
PCL: /usr/lib/x86_64-linux-gnu/libz.so
PCL: /usr/lib/x86_64-linux-gnu/libGLEW.so
PCL: /usr/lib/x86_64-linux-gnu/libSM.so
PCL: /usr/lib/x86_64-linux-gnu/libICE.so
PCL: /usr/lib/x86_64-linux-gnu/libX11.so
PCL: /usr/lib/x86_64-linux-gnu/libXext.so
PCL: /usr/lib/x86_64-linux-gnu/libXt.so
PCL: CMakeFiles/PCL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jaime/Desktop/percepcion_tmp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable PCL"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PCL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PCL.dir/build: PCL

.PHONY : CMakeFiles/PCL.dir/build

CMakeFiles/PCL.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PCL.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PCL.dir/clean

CMakeFiles/PCL.dir/depend:
	cd /home/jaime/Desktop/percepcion_tmp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jaime/Desktop/percepcion_tmp /home/jaime/Desktop/percepcion_tmp /home/jaime/Desktop/percepcion_tmp/build /home/jaime/Desktop/percepcion_tmp/build /home/jaime/Desktop/percepcion_tmp/build/CMakeFiles/PCL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PCL.dir/depend
