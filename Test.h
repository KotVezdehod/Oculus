#pragma once

#ifdef OCULUS_EXPORTS
	#define OCULUS_API __declspec(dllexport)
#else
	#define OCULUS_API __declspec(dllimport)
#endif

extern "C" OCULUS_API void TestFoo();

