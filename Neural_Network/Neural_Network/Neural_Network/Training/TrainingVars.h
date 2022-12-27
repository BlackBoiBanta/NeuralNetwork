#ifdef ML_TRAINING 
	#define NEURONE_T_VARS  float m_value; float m_gradient = 0.0f
	#define NETWORK_T_VARS static uInt times
#else 
	#define NEURONE_T_VARS 
#endif
