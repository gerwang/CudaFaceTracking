#ifndef _CULA_HELPER_H_
#define _CULA_HELPER_H_

#include <cula_sparse.h>

/*safe call macro for CULA API*/
#define culaSafeCall(expr, handle) \
{ \
	culaSparseStatus status = expr; \
	if (expr != culaSparseNoError) { \
		char error_buf[512] = { 0 }; \
		culaSparseGetLastStatusString(handle, error_buf, 512); \
		printf("cula error: %s, in FILE %s, at LINE %d\n", error_buf, __FILE__, __LINE__); \
		std::exit(EXIT_FAILURE); \
	} \
}

#endif