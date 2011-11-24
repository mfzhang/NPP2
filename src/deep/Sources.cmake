include_directories(${NPP2_SOURCE_DIR}/deep)
LIST(APPEND deep_srcs 
	${NPP2_SOURCE_DIR}/deep/DeepAutoEncoder.cpp
	${NPP2_SOURCE_DIR}/deep/NetGenerator.cpp
) 

LIST(APPEND deep_headers
	${NPP2_SOURCE_DIR}/deep/DeepAutoEncoder.h
	${NPP2_SOURCE_DIR}/deep/NetGenerator.h
)