#pragma once

// Public declarations scheduled for removal use this macro and name the
// replacement and earliest removal version in the message.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define GIAO_DEPRECATED(message) [[deprecated(message)]]
#else
#define GIAO_DEPRECATED(message)
#endif
#else
#define GIAO_DEPRECATED(message)
#endif
