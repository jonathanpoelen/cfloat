#include <falcon/cfloat/cfloat.hpp>

template<uint64_t> struct S{};
int main()
{
  using u = uint64_t;
  S<falcon::cfloat::detail::cfloat_add(u(3), u(5))>{};
  S<falcon::cfloat::detail::cfloat_mul(u(3), u(5))>{};
  S<falcon::cfloat::detail::cfloat_sub(u(3), u(5))>{};
  S<falcon::cfloat::detail::cfloat_div(u(3), u(5))>{};
  S<falcon::cfloat::detail::cfloat_div(u(3), u(0))>{};
  S<falcon::cfloat::detail::cfloat_neg(u(3))>{};

  S<falcon::cfloat::detail::cfloat_isgreater(u(3), u(3))>{};
  S<falcon::cfloat::detail::cfloat_isless(u(3), u(3))>{};
  S<falcon::cfloat::detail::cfloat_isgreaterequal(u(3), u(3))>{};
  S<falcon::cfloat::detail::cfloat_islessequal(u(3), u(3))>{};
  S<falcon::cfloat::detail::cfloat_islessgreater(u(3), u(3))>{};
}
