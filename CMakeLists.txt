cmake_minimum_required(VERSION 3.2)
project( lstm )
find_package( OpenCV REQUIRED )
aux_source_directory( src/. src_list )
add_executable( lstm ${src_list} )
target_link_libraries (lstm ${OpenCV_LIBS})

