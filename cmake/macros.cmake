
#
macro(subdirlist result curdir)
  file (GLOB children RELATIVE ${curdir} ${curdir}/*)
  set (dirlist "")
  foreach (child ${children})
    if (IS_DIRECTORY ${curdir}/${child})
      list (APPEND dirlist ${child})
    endif ()
  endforeach ()
  set (${result} ${dirlist})
endmacro ()


# 
macro(get_current_dir_name result)
  get_filename_component(dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  string(REPLACE " " "_" dir ${dir})
  set (${result} ${dir})
endmacro ()