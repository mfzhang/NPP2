include_directories(${NPP2_SOURCE_DIR}/util)
LIST(APPEND util_srcs 
	${NPP2_SOURCE_DIR}/util/LayerRegistry.cpp
	${NPP2_SOURCE_DIR}/util/PatternSet.cpp
	${NPP2_SOURCE_DIR}/util/Registry.cpp
) 

LIST(APPEND util_headers
${NPP2_SOURCE_DIR}/util/LayerRegistry.h
${NPP2_SOURCE_DIR}/util/PatternSet.h
${NPP2_SOURCE_DIR}/util/Registry.h
)