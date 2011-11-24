include_directories(${NPP2_SOURCE_DIR}/core)
LIST(APPEND core_srcs 
	${NPP2_SOURCE_DIR}/core/BasicLayerTypes.cpp
	${NPP2_SOURCE_DIR}/core/FullyConnectedLayer.cpp
	${NPP2_SOURCE_DIR}/core/functions.cpp
	${NPP2_SOURCE_DIR}/core/npp2.cpp
	${NPP2_SOURCE_DIR}/core/NPPException.cpp
) 

LIST(APPEND core_headers
	${NPP2_SOURCE_DIR}/core/BasicLayerTypes.h
	${NPP2_SOURCE_DIR}/core/FullyConnectedLayer.h
	${NPP2_SOURCE_DIR}/core/functions.h
	${NPP2_SOURCE_DIR}/core/npp2.h
	${NPP2_SOURCE_DIR}/core/NPPException.h
)