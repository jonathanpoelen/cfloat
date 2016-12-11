/* The MIT License (MIT)

Copyright (c) 2016 jonathan poelen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
* \author    Jonathan Poelen <jonathan.poelen+falcon@gmail.com>
* \version   0.1
* \brief     64 bits integral type as IEEE 754-2008 floating point
*/

#pragma once

#include <cstdint>
#include <cmath> // FP_NORMAL, FP_SUBNORMAL, FP_ZERO, FP_INFINITE, FP_NAN
#include <type_traits>
#include <limits>

// std::round_indeterminate       Rounding style cannot be determined
// std::round_toward_zero         Rounding toward zero
// std::round_to_nearest          Rounding toward nearest representable value
// std::round_toward_infinity     Rounding toward positive infinity
// std::round_toward_neg_infinity Rounding toward negative infinity

// std::denorm_indeterminate Support of subnormal values cannot be determined
// std::denorm_absent        The type does not support subnormal values
// std::denorm_present       The type allows subnormal values

namespace falcon { namespace cfloat { namespace detail {

template<class Int, class Float, std::size_t SignificandBits>
struct basic_float_traits
{
  using rep_t = Int;
  using fp_t = Float;
  using srep_t = std::make_signed_t<rep_t>;

  static const std::size_t significandBits = SignificandBits;
  static const std::size_t typeWidth       = sizeof(Int) * __CHAR_BIT__;
  static const std::size_t exponentBits    = typeWidth - significandBits - 1;
  static const std::size_t maxExponent     = (1 << exponentBits) - 1;
  static const std::size_t exponentBias    = maxExponent >> 1;

  static const std::size_t implicitBit     = Int(1) << significandBits;
  static const std::size_t significandMask = implicitBit - 1u;
  static const std::size_t signBit         = Int(1) << (significandBits + exponentBits);
  static const std::size_t absMask         = signBit - 1u;
  static const std::size_t exponentMask    = absMask ^ significandMask;
  static const std::size_t oneRep          = exponentBias << significandBits;
  static const std::size_t infRep          = exponentMask;
  static const std::size_t quietBit        = implicitBit >> 1;
  static const std::size_t qnanRep         = exponentMask | quietBit;
};
using float32_traits = basic_float_traits<uint32_t, float, 23>;
using float64_traits = basic_float_traits<uint64_t, double, 52>;

namespace detail
{
  template<class> struct float_traits_from;
  template<> struct float_traits_from<uint32_t> { using type = float32_traits; };
  template<> struct float_traits_from<uint64_t> { using type = float64_traits; };
}

template<class Int>
using float_traits_from = typename detail::float_traits_from<Int>::type;


constexpr int rep_clz(uint32_t a)
{
  return __builtin_clz(a);
}

constexpr int rep_clz(uint64_t a)
{
#if defined __LP64__
  return __builtin_clzl(a);
#else
  if (a & uint64_t(0xffffffff00000000))
    return __builtin_clz(a >> 32);
  else
    return 32 + __builtin_clz(a & uint64_t(0xffffffff));
#endif
}

constexpr int rep_clz(long long a)
{
#if defined __LP64__
  return __builtin_clzl(a);
#else
  if (a & uint64_t(0xffffffff00000000))
    return __builtin_clz(a >> 32);
  else
    return 32 + __builtin_clz(a & uint64_t(0xffffffff));
#endif
}

// 32x32 --> 64 bit multiply
constexpr void wideMultiply(uint32_t a, uint32_t b, uint32_t *hi, uint32_t *lo)
{
    const uint64_t product = uint64_t(a)*b;
    *hi = uint32_t(product >> 32);
    *lo = uint32_t(product);
}

constexpr uint64_t loWord(uint64_t a)
{
  return a & 0xffffffffU;
}

constexpr uint64_t hiWord(uint64_t a)
{
  return a >> 32;
}

// 64x64 -> 128 wide multiply for platforms that don't have such an operation;
// many 64-bit platforms have this operation, but they tend to have hardware
// floating-point, so we don't bother with a special case for them here.
constexpr void wideMultiply(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
  // Each of the component 32x32 -> 64 products
  const uint64_t plolo = loWord(a) * loWord(b);
  const uint64_t plohi = loWord(a) * hiWord(b);
  const uint64_t philo = hiWord(a) * loWord(b);
  const uint64_t phihi = hiWord(a) * hiWord(b);
  // Sum terms that contribute to lo in a way that allows us to get the carry
  const uint64_t r0 = loWord(plolo);
  const uint64_t r1 = hiWord(plolo) + loWord(plohi) + loWord(philo);
  *lo = r0 + (r1 << 32);
  // Sum terms contributing to hi with the carry from lo
  *hi = hiWord(plohi) + hiWord(philo) + hiWord(r1) + phihi;
}


template<class Int>
constexpr int normalize(Int * significand)
{
  using cons = float_traits_from<Int>;
  const int shift = rep_clz(*significand) - rep_clz(cons::implicitBit);
  *significand <<= shift;
  return 1 - shift;
}

template<class Int>
constexpr void wideLeftShift(Int * hi, Int * lo, int count)
{
  using cons = float_traits_from<Int>;
  *hi = *hi << count | *lo >> (cons::typeWidth - count);
  *lo = *lo << count;
}

template<class Int>
constexpr void wideRightShiftWithSticky(Int *hi, Int *lo, unsigned count)
{
  using cons = float_traits_from<Int>;

  if (count < cons::typeWidth) {
    const bool sticky = *lo << (cons::typeWidth - count);
    *lo = *hi << (cons::typeWidth - count) | *lo >> count | sticky;
    *hi = *hi >> count;
  }
  else if (count < 2 * cons::typeWidth) {
    const bool sticky = *hi << (2 * cons::typeWidth - count) | *lo;
    *lo = *hi >> (count - cons::typeWidth) | sticky;
    *hi = 0;
  } else {
    const bool sticky = *hi | *lo;
    *lo = sticky;
    *hi = 0;
  }
}

template<class Int>
constexpr Int cfloat_add(Int a, Int b)
{
  using cons = float_traits_from<Int>;
  using rep_t = typename cons::rep_t;

  rep_t aRep = a;
  rep_t bRep = b;
  const rep_t aAbs = aRep & cons::absMask;
  const rep_t bAbs = bRep & cons::absMask;

  // Detect if a or b is zero, infinity, or NaN.
  if (aAbs - rep_t(1) >= cons::infRep - rep_t(1) ||
    bAbs - rep_t(1) >= cons::infRep - rep_t(1)) {
    // NaN + anything = qNaN
    if (aAbs > cons::infRep) return a | cons::quietBit;
    // anything + NaN = qNaN
    if (bAbs > cons::infRep) return b | cons::quietBit;

    if (aAbs == cons::infRep) {
      // +/-infinity + -/+infinity = qNaN
      if ((a ^ b) == cons::signBit) return cons::qnanRep;
      // +/-infinity + anything remaining = +/- infinity
      else return a;
    }

    // anything remaining + +/-infinity = +/-infinity
    if (bAbs == cons::infRep) return b;

    // zero + anything = anything
    if (!aAbs) {
      // but we need to get the sign right for zero + zero
      if (!bAbs) return a & b;
      else return b;
    }

    // anything + zero = anything
    if (!bAbs) return a;
  }

  // Swap a and b if necessary so that a has the larger absolute value.
  if (bAbs > aAbs) {
    const rep_t temp = aRep;
    aRep = bRep;
    bRep = temp;
  }

  // Extract the exponent and significand from the (possibly swapped) a and b.
  int aExponent = int(aRep >> cons::significandBits & cons::maxExponent);
  int bExponent = int(bRep >> cons::significandBits & cons::maxExponent);
  rep_t aSignificand = aRep & cons::significandMask;
  rep_t bSignificand = bRep & cons::significandMask;

  // Normalize any denormals, and adjust the exponent accordingly.
  if (aExponent == 0) aExponent = normalize(&aSignificand);
  if (bExponent == 0) bExponent = normalize(&bSignificand);

  // The sign of the result is the sign of the larger operand, a.  If they
  // have opposite signs, we are performing a subtraction; otherwise addition.
  const rep_t resultSign = aRep & cons::signBit;
  const bool subtraction = (aRep ^ bRep) & cons::signBit;

  // Shift the significands to give us round, guard and sticky, and or in the
  // implicit significand bit.  (If we fell through from the denormal path it
  // was already set by normalize( ), but setting it twice won't hurt
  // anything.)
  aSignificand = (aSignificand | cons::implicitBit) << 3;
  bSignificand = (bSignificand | cons::implicitBit) << 3;

  // Shift the significand of b by the difference in exponents, with a sticky
  // bottom bit to get rounding correct.
  const unsigned align = unsigned(aExponent - bExponent);
  if (align) {
    if (align < cons::typeWidth) {
      const bool sticky = bSignificand << (cons::typeWidth - align);
      bSignificand = bSignificand >> align | sticky;
    } else {
      bSignificand = 1; // sticky; b is known to be non-zero.
    }
  }
  if (subtraction) {
    aSignificand -= bSignificand;
    // If a == -b, return +zero.
    if (aSignificand == 0) return 0;

    // If partial cancellation occured, we need to left-shift the result
    // and adjust the exponent:
    if (aSignificand < cons::implicitBit << 3) {
      const int shift = rep_clz(aSignificand) - rep_clz(cons::implicitBit << 3);
      aSignificand <<= shift;
      aExponent -= shift;
    }
  }
  else /* addition */ {
    aSignificand += bSignificand;

    // If the addition carried up, we need to right-shift the result and
    // adjust the exponent:
    if (aSignificand & cons::implicitBit << 4) {
      const bool sticky = aSignificand & 1;
      aSignificand = aSignificand >> 1 | sticky;
      aExponent += 1;
    }
  }

  // If we have overflowed the type, return +/- infinity:
  if (aExponent >= int{cons::maxExponent}) return cons::infRep | resultSign;

  if (aExponent <= 0) {
    // Result is denormal before rounding; the exponent is zero and we
    // need to shift the significand.
    const int shift = 1 - aExponent;
    const bool sticky = aSignificand << (int(cons::typeWidth) - shift);
    aSignificand = aSignificand >> shift | sticky;
    aExponent = 0;
  }

  // Low three bits are round, guard, and sticky.
  const int roundGuardSticky = aSignificand & 0x7;

  // Shift the significand into place, and mask off the implicit bit.
  rep_t result = aSignificand >> 3 & cons::significandMask;

  // Insert the exponent and sign.
  result |= rep_t(aExponent) << cons::significandBits;
  result |= resultSign;

  // Final rounding.  The result may overflow to infinity, but that is the
  // correct result in that case.
  if (roundGuardSticky > 0x4) result++;
  if (roundGuardSticky == 0x4) result += result & 1;
  return result;
}

template<class Int>
constexpr Int cfloat_mul(Int a, Int b)
{
  using cons = float_traits_from<Int>;
  using rep_t = typename cons::rep_t;

  const unsigned aExponent = a >> cons::significandBits & cons::maxExponent;
  const unsigned bExponent = b >> cons::significandBits & cons::maxExponent;
  const rep_t productSign = (a ^ b) & cons::signBit;

  rep_t aSignificand = a & cons::significandMask;
  rep_t bSignificand = b & cons::significandMask;
  int scale = 0;

  // Detect if a or b is zero, denormal, infinity, or NaN.
  if (aExponent-1U >= cons::maxExponent-1U || bExponent-1U >= cons::maxExponent-1U) {
    const rep_t aAbs = a & cons::absMask;
    const rep_t bAbs = b & cons::absMask;

    // NaN * anything = qNaN
    if (aAbs > cons::infRep) return a | cons::quietBit;
    // anything * NaN = qNaN
    if (bAbs > cons::infRep) return b | cons::quietBit;

    if (aAbs == cons::infRep) {
      // infinity * non-zero = +/- infinity
      if (bAbs) return aAbs | productSign;
      // infinity * zero = NaN
      else return cons::qnanRep;
    }

    if (bAbs == cons::infRep) {
      //? non-zero * infinity = +/- infinity
      if (aAbs) return bAbs | productSign;
      // zero * infinity = NaN
      else return cons::qnanRep;
    }

    // zero * anything = +/- zero
    if (!aAbs) return productSign;
    // anything * zero = +/- zero
    if (!bAbs) return productSign;

    // one or both of a or b is denormal, the other (if applicable) is a
    // normal number.  Renormalize one or both of a and b, and set scale to
    // include the necessary exponent adjustment.
    if (aAbs < cons::implicitBit) scale += normalize(&aSignificand);
    if (bAbs < cons::implicitBit) scale += normalize(&bSignificand);
  }

  // Or in the implicit significand bit.  (If we fell through from the
  // denormal path it was already set by normalize( ), but setting it twice
  // won't hurt anything.)
  aSignificand |= cons::implicitBit;
  bSignificand |= cons::implicitBit;

  // Get the significand of a*b.  Before multiplying the significands, shift
  // one of them left to left-align it in the field.  Thus, the product will
  // have (exponentBits + 2) integral digits, all but two of which must be
  // zero.  Normalizing this result is just a conditional left-shift by one
  // and bumping the exponent accordingly.
  rep_t productHi = 0, productLo = 0;
  wideMultiply(aSignificand, bSignificand << cons::exponentBits,
                &productHi, &productLo);

  int productExponent = int(aExponent + bExponent - cons::exponentBias + scale);

  // Normalize the significand, adjust exponent if needed.
  if (productHi & cons::implicitBit) productExponent++;
  else wideLeftShift(&productHi, &productLo, 1);

  // If we have overflowed the type, return +/- infinity.
  if (productExponent >= int{cons::maxExponent}) return cons::infRep | productSign;

  if (productExponent <= 0) {
    // Result is denormal before rounding
    //
    // If the result is so small that it just underflows to zero, return
    // a zero of the appropriate sign.  Mathematically there is no need to
    // handle this case separately, but we make it a special case to
    // simplify the shift logic.
    const unsigned shift = unsigned(rep_t(1) - unsigned(productExponent));
    if (shift >= cons::typeWidth) return productSign;

    // Otherwise, shift the significand of the result so that the round
    // bit is the high bit of productLo.
    wideRightShiftWithSticky(&productHi, &productLo, shift);
  }
  else {
    // Result is normal before rounding; insert the exponent.
    productHi &= cons::significandMask;
    productHi |= rep_t(productExponent) << cons::significandBits;
  }

  // Insert the sign of the result:
  productHi |= productSign;

  // Final rounding.  The final result may overflow to infinity, or underflow
  // to zero, but those are the correct results in those cases.  We use the
  // default IEEE-754 round-to-nearest, ties-to-even rounding mode.
  if (productLo > cons::signBit) productHi++;
  if (productLo == cons::signBit) productHi += productHi & 1;
  return productHi;
}

template<class Int>
constexpr Int cfloat_neg(Int a)
{
  using cons = float_traits_from<Int>;
  return a ^ cons::signBit;
}

template<class Int>
constexpr Int cfloat_sub(Int a, Int b)
{
  return cfloat_add(a, cfloat_neg(b));
}

constexpr uint64_t cfloat_div(uint64_t a, uint64_t b)
{
  using cons = float_traits_from<uint64_t>;
  using rep_t = uint64_t;

  const unsigned aExponent = a >> cons::significandBits & cons::maxExponent;
  const unsigned bExponent = b >> cons::significandBits & cons::maxExponent;
  const rep_t quotientSign = (a ^ b) & cons::signBit;

  rep_t aSignificand = a & cons::significandMask;
  rep_t bSignificand = b & cons::significandMask;
  int scale = 0;

  // Detect if a or b is zero, denormal, infinity, or NaN.
  if (aExponent-1U >= cons::maxExponent-1U || bExponent-1U >= cons::maxExponent-1U) {
    const rep_t aAbs = a & cons::absMask;
    const rep_t bAbs = b & cons::absMask;

    // NaN / anything = qNaN
    if (aAbs > cons::infRep) return a | cons::quietBit;
    // anything / NaN = qNaN
    if (bAbs > cons::infRep) return b | cons::quietBit;

    if (aAbs == cons::infRep) {
      // infinity / infinity = NaN
      if (bAbs == cons::infRep) return cons::qnanRep;
      // infinity / anything else = +/- infinity
      else return aAbs | quotientSign;
    }

    // anything else / infinity = +/- 0
    if (bAbs == cons::infRep) return quotientSign;

    if (!aAbs) {
      // zero / zero = NaN
      if (!bAbs) return cons::qnanRep;
      // zero / anything else = +/- zero
      else return quotientSign;
    }
    // anything else / zero = +/- infinity
    if (!bAbs) return cons::infRep | quotientSign;

    // one or both of a or b is denormal, the other (if applicable) is a
    // normal number.  Renormalize one or both of a and b, and set scale to
    // include the necessary exponent adjustment.
    if (aAbs < cons::implicitBit) scale += normalize(&aSignificand);
    if (bAbs < cons::implicitBit) scale -= normalize(&bSignificand);
  }

  // Or in the implicit significand bit.  (If we fell through from the
  // denormal path it was already set by normalize( ), but setting it twice
  // won't hurt anything.)
  aSignificand |= cons::implicitBit;
  bSignificand |= cons::implicitBit;
  int quotientExponent = aExponent - bExponent + scale;

  // Align the significand of b as a Q31 fixed-point number in the range
  // [1, 2.0) and get a Q32 approximate reciprocal using a small minimax
  // polynomial approximation: reciprocal = 3/4 + 1/sqrt(2) - b/2.  This
  // is accurate to about 3.5 binary digits.
  const uint32_t q31b = uint32_t(bSignificand >> 21);
  uint32_t recip32 = UINT32_C(0x7504f333) - q31b;

  // Now refine the reciprocal estimate using a Newton-Raphson iteration:
  //
  //     x1 = x0 * (2 - x0 * b)
  //
  // This doubles the number of correct binary digits in the approximation
  // with each iteration, so after three iterations, we have about 28 binary
  // digits of accuracy.
  uint32_t correction32 = 0;
  correction32 = uint32_t(-(uint64_t(recip32) * q31b >> 32));
  recip32 = uint32_t(uint64_t(recip32) * correction32 >> 31);
  correction32 = uint32_t(-(uint64_t(recip32) * q31b >> 32));
  recip32 = uint32_t(uint64_t(recip32) * correction32 >> 31);
  correction32 = uint32_t(-(uint64_t(recip32) * q31b >> 32));
  recip32 = uint32_t(uint64_t(recip32) * correction32 >> 31);

  // recip32 might have overflowed to exactly zero in the preceding
  // computation if the high word of b is exactly 1.0.  This would sabotage
  // the full-width final stage of the computation that follows, so we adjust
  // recip32 downward by one bit.
  recip32--;

  // We need to perform one more iteration to get us to 56 binary digits;
  // The last iteration needs to happen with extra precision.
  const uint32_t q63blo = uint32_t(bSignificand << 11);
  uint64_t correction = -(uint64_t(recip32)*q31b + (uint64_t(recip32)*q63blo >> 32));
  uint32_t cHi = uint32_t(correction >> 32);
  uint32_t cLo = uint32_t(correction);
  uint64_t reciprocal = uint64_t(recip32)*cHi + (uint64_t(recip32)*cLo >> 32);

  // We already adjusted the 32-bit estimate, now we need to adjust the final
  // 64-bit reciprocal estimate downward to ensure that it is strictly smaller
  // than the infinitely precise exact reciprocal.  Because the computation
  // of the Newton-Raphson step is truncating at every step, this adjustment
  // is small; most of the work is already done.
  reciprocal -= 2;

  // The numerical reciprocal is accurate to within 2^-56, lies in the
  // interval [0.5, 1.0), and is strictly smaller than the true reciprocal
  // of b.  Multiplying a by this reciprocal thus gives a numerical q = a/b
  // in Q53 with the following properties:
  //
  //    1. q < a/b
  //    2. q is in the interval [0.5, 2.0)
  //    3. the error in q is bounded away from 2^-53 (actually, we have a
  //       couple of bits to spare, but this is all we need).

  // We need a 64 x 64 multiply high to compute q, which isn't a basic
  // operation in C, so we need to be a little bit fussy.
  rep_t quotient = 0, quotientLo = 0;
  wideMultiply(aSignificand << 2, reciprocal, &quotient, &quotientLo);

  // Two cases: quotient is in [0.5, 1.0) or quotient is in [1.0, 2.0).
  // In either case, we are going to compute a residual of the form
  //
  //     r = a - q*b
  //
  // We know from the construction of q that r satisfies:
  //
  //     0 <= r < ulp(q)*b
  //
  // if r is greater than 1/2 ulp(q)*b, then q rounds up.  Otherwise, we
  // already have the correct result.  The exact halfway case cannot occur.
  // We also take this time to right shift quotient if it falls in the [1,2)
  // range and adjust the exponent accordingly.
  rep_t residual = 0;

  if (quotient < (cons::implicitBit << 1)) {
    residual = (aSignificand << 53) - quotient * bSignificand;
    quotientExponent--;
  } else {
    quotient >>= 1;
    residual = (aSignificand << 52) - quotient * bSignificand;
  }

  const int writtenExponent = int(quotientExponent + cons::exponentBias);

  if (writtenExponent >= int{cons::maxExponent}) {
    // If we have overflowed the exponent, return infinity.
    return cons::infRep | quotientSign;
  }
  else if (writtenExponent < 1) {
    // Flush denormals to zero.  In the future, it would be nice to add
    // code to round them correctly.
    return quotientSign;
  }
  else {
    const bool round = (residual << 1) > bSignificand;
    // Clear the implicit bit
    rep_t absResult = quotient & cons::significandMask;
    // Insert the exponent
    absResult |= rep_t(writtenExponent) << cons::significandBits;
    // Round
    absResult += round;
    // Insert the sign and return
    return absResult | quotientSign;
  }
}

constexpr uint32_t cfloat_div(uint32_t a, uint32_t b)
{
  using cons = float_traits_from<uint32_t>;
  using rep_t = uint32_t;

  const unsigned aExponent = a >> cons::significandBits & cons::maxExponent;
  const unsigned bExponent = b >> cons::significandBits & cons::maxExponent;
  const rep_t quotientSign = (a ^ b) & cons::signBit;

  rep_t aSignificand = a & cons::significandMask;
  rep_t bSignificand = b & cons::significandMask;
  int scale = 0;

  // Detect if a or b is zero, denormal, infinity, or NaN.
  if (aExponent-1U >= cons::maxExponent-1U || bExponent-1U >= cons::maxExponent-1U) {
    const rep_t aAbs = a & cons::absMask;
    const rep_t bAbs = b & cons::absMask;

    // NaN / anything = qNaN
    if (aAbs > cons::infRep) return a | cons::quietBit;
    // anything / NaN = qNaN
    if (bAbs > cons::infRep) return b | cons::quietBit;

    if (aAbs == cons::infRep) {
      // infinity / infinity = NaN
      if (bAbs == cons::infRep) return cons::qnanRep;
      // infinity / anything else = +/- infinity
      else return aAbs | quotientSign;
    }

    // anything else / infinity = +/- 0
    if (bAbs == cons::infRep) return quotientSign;

    if (!aAbs) {
      // zero / zero = NaN
      if (!bAbs) return cons::qnanRep;
      // zero / anything else = +/- zero
      else return quotientSign;
    }
    // anything else / zero = +/- infinity
    if (!bAbs) return cons::infRep | quotientSign;

    // one or both of a or b is denormal, the other (if applicable) is a
    // normal number.  Renormalize one or both of a and b, and set scale to
    // include the necessary exponent adjustment.
    if (aAbs < cons::implicitBit) scale += normalize(&aSignificand);
    if (bAbs < cons::implicitBit) scale -= normalize(&bSignificand);
  }

  // Or in the implicit significand bit.  (If we fell through from the
  // denormal path it was already set by normalize( ), but setting it twice
  // won't hurt anything.)
  aSignificand |= rep_t{cons::implicitBit};
  bSignificand |= rep_t{cons::implicitBit};
  int quotientExponent = aExponent - bExponent + scale;

  // Align the significand of b as a Q31 fixed-point number in the range
  // [1, 2.0) and get a Q32 approximate reciprocal using a small minimax
  // polynomial approximation: reciprocal = 3/4 + 1/sqrt(2) - b/2.  This
  // is accurate to about 3.5 binary digits.
  uint32_t q31b = bSignificand << 8;
  uint32_t reciprocal = UINT32_C(0x7504f333) - q31b;

  // Now refine the reciprocal estimate using a Newton-Raphson iteration:
  //
  //     x1 = x0 * (2 - x0 * b)
  //
  // This doubles the number of correct binary digits in the approximation
  // with each iteration, so after three iterations, we have about 28 binary
  // digits of accuracy.
  uint32_t correction = 0;
  correction = uint32_t(-(uint64_t(reciprocal) * q31b >> 32));
  reciprocal = uint32_t(uint64_t(reciprocal) * correction >> 31);
  correction = uint32_t(-(uint64_t(reciprocal) * q31b >> 32));
  reciprocal = uint32_t(uint64_t(reciprocal) * correction >> 31);
  correction = uint32_t(-(uint64_t(reciprocal) * q31b >> 32));
  reciprocal = uint32_t(uint64_t(reciprocal) * correction >> 31);

  // Exhaustive testing shows that the error in reciprocal after three steps
  // is in the interval [-0x1.f58108p-31, 0x1.d0e48cp-29], in line with our
  // expectations.  We bump the reciprocal by a tiny value to force the error
  // to be strictly positive (in the range [0x1.4fdfp-37,0x1.287246p-29], to
  // be specific).  This also causes 1/1 to give a sensible approximation
  // instead of zero (due to overflow).
  reciprocal -= 2;

  // The numerical reciprocal is accurate to within 2^-28, lies in the
  // interval [0x1.000000eep-1, 0x1.fffffffcp-1], and is strictly smaller
  // than the true reciprocal of b.  Multiplying a by this reciprocal thus
  // gives a numerical q = a/b in Q24 with the following properties:
  //
  //    1. q < a/b
  //    2. q is in the interval [0x1.000000eep-1, 0x1.fffffffcp0)
  //    3. the error in q is at most 2^-24 + 2^-27 -- the 2^24 term comes
  //       from the fact that we truncate the product, and the 2^27 term
  //       is the error in the reciprocal of b scaled by the maximum
  //       possible value of a.  As a consequence of this error bound,
  //       either q or nextafter(q) is the correctly rounded
  rep_t quotient = rep_t(uint64_t(reciprocal)*(aSignificand << 1) >> 32);

  // Two cases: quotient is in [0.5, 1.0) or quotient is in [1.0, 2.0).
  // In either case, we are going to compute a residual of the form
  //
  //     r = a - q*b
  //
  // We know from the construction of q that r satisfies:
  //
  //     0 <= r < ulp(q)*b
  //
  // if r is greater than 1/2 ulp(q)*b, then q rounds up.  Otherwise, we
  // already have the correct result.  The exact halfway case cannot occur.
  // We also take this time to right shift quotient if it falls in the [1,2)
  // range and adjust the exponent accordingly.
  rep_t residual = 0;

  if (quotient < (cons::implicitBit << 1)) {
    residual = (aSignificand << 24) - quotient * bSignificand;
    quotientExponent--;
  } else {
    quotient >>= 1;
    residual = (aSignificand << 23) - quotient * bSignificand;
  }

  const int writtenExponent = quotientExponent + int{cons::exponentBias};

  if (writtenExponent >= int{cons::maxExponent}) {
    // If we have overflowed the exponent, return infinity.
    return cons::infRep | quotientSign;
  }
  else if (writtenExponent < 1) {
    // Flush denormals to zero.  In the future, it would be nice to add
    // code to round them correctly.
    return quotientSign;
  }
  else {
    const bool round = (residual << 1) > bSignificand;
    // Clear the implicit bit
    rep_t absResult = quotient & cons::significandMask;
    // Insert the exponent
    absResult |= rep_t(writtenExponent) << cons::significandBits;
    // Round
    absResult += round;
    // Insert the sign and return
    return absResult | quotientSign;
  }
}

constexpr uint64_t cfloat_extend(uint32_t a)
{
  using src_t = uint32_t;
  using src_rep_t = uint32_t;
  using dst_t = uint64_t;
  using dst_rep_t = uint64_t;
  constexpr auto srcSigBits = float32_traits::significandBits;
  constexpr auto dstSigBits = float64_traits::significandBits;

  // Various constants whose values follow from the type parameters.
  // Any reasonable optimizer will fold and propagate all of these.
  const int srcBits = sizeof(src_t)*__CHAR_BIT__;
  const int srcExpBits = srcBits - srcSigBits - 1;
  const int srcInfExp = (1 << srcExpBits) - 1;
  const int srcExpBias = srcInfExp >> 1;

  const src_rep_t srcMinNormal = src_rep_t(1) << srcSigBits;
  const src_rep_t srcInfinity = src_rep_t(srcInfExp) << srcSigBits;
  const src_rep_t srcSignMask = src_rep_t(1) << (srcSigBits + srcExpBits);
  const src_rep_t srcAbsMask = srcSignMask - 1;
  const src_rep_t srcQNaN = src_rep_t(1) << (srcSigBits - 1);
  const src_rep_t srcNaNCode = srcQNaN - 1;

  const int dstBits = sizeof(dst_t)*__CHAR_BIT__;
  const int dstExpBits = dstBits - dstSigBits - 1;
  const int dstInfExp = (1 << dstExpBits) - 1;
  const int dstExpBias = dstInfExp >> 1;

  const dst_rep_t dstMinNormal = dst_rep_t(1) << dstSigBits;

  // Break a into a sign and representation of the absolute value
  const src_rep_t aRep = a;
  const src_rep_t aAbs = aRep & srcAbsMask;
  const src_rep_t sign = aRep & srcSignMask;
  dst_rep_t absResult = 0;

  // If sizeof(src_rep_t) < sizeof(int), the subtraction result is promoted
  // to (signed) int.  To avoid that, explicitly cast to src_rep_t.
  if (src_rep_t(aAbs - srcMinNormal) < srcInfinity - srcMinNormal) {
    // a is a normal number.
    // Extend to the destination type by shifting the significand and
    // exponent into the proper position and rebiasing the exponent.
    absResult = dst_rep_t(aAbs) << (dstSigBits - srcSigBits);
    absResult += dst_rep_t(dstExpBias - srcExpBias) << dstSigBits;
  }

  else if (aAbs >= srcInfinity) {
    // a is NaN or infinity.
    // Conjure the result by beginning with infinity, then setting the qNaN
    // bit (if needed) and right-aligning the rest of the trailing NaN
    // payload field.
    absResult = dst_rep_t(dstInfExp) << dstSigBits;
    absResult |= dst_rep_t(aAbs & srcQNaN) << (dstSigBits - srcSigBits);
    absResult |= dst_rep_t(aAbs & srcNaNCode) << (dstSigBits - srcSigBits);
  }

  else if (aAbs) {
    // a is denormal.
    // renormalize the significand and clear the leading bit, then insert
    // the correct adjusted exponent in the destination type.
    const int scale = rep_clz(aAbs) - rep_clz(srcMinNormal);
    absResult = dst_rep_t(aAbs) << (dstSigBits - srcSigBits + scale);
    absResult ^= dstMinNormal;
    const int resultExponent = dstExpBias - srcExpBias - scale + 1;
    absResult |= dst_rep_t(resultExponent) << dstSigBits;
  }
  else {
    // a is zero.
    absResult = 0;
  }

  // Apply the signbit to (dst_t)abs(a).
  return absResult | dst_rep_t(sign) << (dstBits - srcBits);
}

constexpr uint32_t cfloat_trunc(uint64_t a)
{
  using src_t = uint64_t;
  using src_rep_t = uint64_t;
  using dst_t = uint32_t;
  using dst_rep_t = uint32_t;
  constexpr auto srcSigBits = float64_traits::significandBits;
  constexpr auto dstSigBits = float32_traits::significandBits;

  // Various constants whose values follow from the type parameters.
  // Any reasonable optimizer will fold and propagate all of these.
  const int srcBits = sizeof(src_t)*__CHAR_BIT__;
  const int srcExpBits = srcBits - srcSigBits - 1;
  const int srcInfExp = (1 << srcExpBits) - 1;
  const int srcExpBias = srcInfExp >> 1;

  const src_rep_t srcMinNormal = src_rep_t(1) << srcSigBits;
  const src_rep_t srcSignificandMask = srcMinNormal - 1;
  const src_rep_t srcInfinity = src_rep_t(srcInfExp) << srcSigBits;
  const src_rep_t srcSignMask = src_rep_t(1) << (srcSigBits + srcExpBits);
  const src_rep_t srcAbsMask = srcSignMask - 1;
  const src_rep_t roundMask = (src_rep_t(1) << (srcSigBits - dstSigBits)) - 1;
  const src_rep_t halfway = src_rep_t(1) << (srcSigBits - dstSigBits - 1);
  const src_rep_t srcQNaN = src_rep_t(1) << (srcSigBits - 1);
  const src_rep_t srcNaNCode = srcQNaN - 1;

  const int dstBits = sizeof(dst_t)*__CHAR_BIT__;
  const int dstExpBits = dstBits - dstSigBits - 1;
  const int dstInfExp = (1 << dstExpBits) - 1;
  const int dstExpBias = dstInfExp >> 1;

  const int underflowExponent = srcExpBias + 1 - dstExpBias;
  const int overflowExponent = srcExpBias + dstInfExp - dstExpBias;
  const src_rep_t underflow = src_rep_t(underflowExponent) << srcSigBits;
  const src_rep_t overflow = src_rep_t(overflowExponent) << srcSigBits;

  const dst_rep_t dstQNaN = dst_rep_t(1) << (dstSigBits - 1);
  const dst_rep_t dstNaNCode = dstQNaN - 1;

  // Break a into a sign and representation of the absolute value
  const src_rep_t aRep = a;
  const src_rep_t aAbs = aRep & srcAbsMask;
  const src_rep_t sign = aRep & srcSignMask;
  dst_rep_t absResult = 0;

  if (aAbs - underflow < aAbs - overflow) {
    // The exponent of a is within the range of normal numbers in the
    // destination format.  We can convert by simply right-shifting with
    // rounding and adjusting the exponent.
    absResult = dst_rep_t(aAbs >> (srcSigBits - dstSigBits));
    absResult -= dst_rep_t(srcExpBias - dstExpBias) << dstSigBits;

    const src_rep_t roundBits = aAbs & roundMask;
    // Round to nearest
    if (roundBits > halfway)
      absResult++;
    // Ties to even
    else if (roundBits == halfway)
      absResult += absResult & 1;
  }
  else if (aAbs > srcInfinity) {
    // a is NaN.
    // Conjure the result by beginning with infinity, setting the qNaN
    // bit and inserting the (truncated) trailing NaN field.
    absResult = dst_rep_t(dstInfExp) << dstSigBits;
    absResult |= dstQNaN;
    absResult |= dst_rep_t(((aAbs & srcNaNCode) >> (srcSigBits - dstSigBits)) & dstNaNCode);
  }
  else if (aAbs >= overflow) {
    // a overflows to infinity.
    absResult = dst_rep_t(dstInfExp) << dstSigBits;
  }
  else {
    // a underflows on conversion to the destination type or is an exact
    // zero.  The result may be a denormal or zero.  Extract the exponent
    // to get the shift amount for the denormalization.
    const int aExp = int(aAbs >> srcSigBits);
    const int shift = srcExpBias - dstExpBias - aExp + 1;

    const src_rep_t significand = (aRep & srcSignificandMask) | srcMinNormal;

    // Right shift by the denormalization amount with sticky.
    if (shift > int{srcSigBits}) {
      absResult = 0;
    } else {
      const bool sticky = significand << (srcBits - shift);
      src_rep_t denormalizedSignificand = significand >> shift | sticky;
      absResult = dst_rep_t(denormalizedSignificand >> (srcSigBits - dstSigBits));
      const src_rep_t roundBits = denormalizedSignificand & roundMask;
      // Round to nearest
      if (roundBits > halfway)
        absResult++;
      // Ties to even
      else if (roundBits == halfway)
        absResult += absResult & 1;
    }
  }

  // Apply the signbit to (dst_t)abs(a).
  return dst_rep_t(absResult | sign >> (srcBits - dstBits));
}

template<class fixint_t, class Int>
constexpr fixint_t cfloat_fixint(Int a)
{
  using fixuint_t = std::make_unsigned_t<fixint_t>;
  using cons = float_traits_from<Int>;
  using rep_t = typename cons::rep_t;

  const fixint_t fixint_max = fixint_t((~fixuint_t(0)) / 2);
  const fixint_t fixint_min = -fixint_max - 1;
  // Break a into sign, exponent, significand
  const rep_t aRep = a;
  const rep_t aAbs = aRep & cons::absMask;
  const fixint_t sign = aRep & cons::signBit ? -1 : 1;
  const int exponent = (aAbs >> cons::significandBits) - cons::exponentBias;
  const rep_t significand = (aAbs & cons::significandMask) | cons::implicitBit;

  // If exponent is negative, the result is zero.
  if (exponent < 0)
    return 0;

  // If the value is too large for the integer type, saturate.
  if (unsigned(exponent) >= sizeof(fixint_t) * __CHAR_BIT__)
    return sign == 1 ? cons::fixint_max : cons::fixint_min;

  // If 0 <= exponent < significandBits, right shift to get the result.
  // Otherwise, shift left.
  if (exponent < cons::significandBits)
    return sign * (significand >> (cons::significandBits - exponent));
  else
    return sign * (fixint_t(significand) << (exponent - cons::significandBits));
}

template<class fixuint_t, class Int>
constexpr fixuint_t cfloat_fixuint(Int a)
{
  using cons = float_traits_from<Int>;
  using rep_t = typename cons::rep_t;

  // Break a into sign, exponent, significand
  const rep_t aRep = a;
  const rep_t aAbs = aRep & cons::absMask;
  const int sign = aRep & cons::signBit ? -1 : 1;
  const int exponent = (aAbs >> cons::significandBits) - cons::exponentBias;
  const rep_t significand = (aAbs & cons::significandMask) | cons::implicitBit;

  // If either the value or the exponent is negative, the result is zero.
  if (sign == -1 || exponent < 0)
    return 0;

  // If the value is too large for the integer type, saturate.
  if (unsigned(exponent) > sizeof(fixuint_t) * __CHAR_BIT__)
    return ~fixuint_t(0);

  // If 0 <= exponent < significandBits, right shift to get the result.
  // Otherwise, shift left.
  if (exponent < cons::significandBits)
    return significand >> (cons::significandBits - exponent);
  else
    return fixuint_t(significand) << (exponent - cons::significandBits);
}

template<class float_traits, class int_or_unsigned>
constexpr typename float_traits::rep_t
cfloat_i2f(int_or_unsigned a)
{
  static_assert(
    std::is_same<int_or_unsigned, int>::value ||
    std::is_same<int_or_unsigned, unsigned>::value,
    "int or unsigned only"
  );
  using cons = float_traits;
  using rep_t = typename float_traits::rep_t;

  const int aWidth = sizeof a * __CHAR_BIT__;

  // Handle zero as a special case to protect clz
  if (a == 0)
    return 0;

  // All other cases begin by extracting the sign and absolute value of a
  rep_t sign = 0;
  if (std::is_same<int_or_unsigned, int>::value && a < 0) {
    sign = cons::signBit;
    a = -a;
  }

  // Exponent of (fp_t)a is the width of abs(a).
  const int exponent = (aWidth - 1) - rep_clz(a);
  rep_t result;

  static_assert(
    std::is_same<uint32_t, rep_t>::value ||
    std::is_same<uint64_t, rep_t>::value,
    "uint32_t or uint64_t only");
  if (std::is_same<uint64_t, rep_t>::value) {
    // Shift a into the significand field and clear the implicit bit.  Extra
    // cast to unsigned int is necessary to get the correct behavior for
    // the input INT_MIN.
    const int shift = cons::significandBits - exponent;
    result = rep_t(unsigned(a)) << shift ^ cons::implicitBit;
  }
  else {
    // Shift a into the significand field, rounding if it is a right-shift
    if (exponent <= cons::significandBits) {
      const int shift = cons::significandBits - exponent;
      result = rep_t(a) << shift ^ cons::implicitBit;
    } else {
      const int shift = exponent - cons::significandBits;
      result = rep_t(a) >> shift ^ cons::implicitBit;
      rep_t round = rep_t(a) << (cons::typeWidth - shift);
      if (round > cons::signBit) result++;
      if (round == cons::signBit) result += result & 1;
    }
  }

  // Insert the exponent
  result += rep_t(exponent + cons::exponentBias) << cons::significandBits;
  // Insert the sign bit and return
  return result | sign;
}

template<class float_traits, class ll_or_ull>
constexpr typename float_traits::rep_t
clfoat_ll2f(ll_or_ull a)
{
  static_assert(
    std::is_same<ll_or_ull, long long>::value ||
    std::is_same<ll_or_ull, unsigned long long>::value,
    "long long or unsigned long long only"
  );
  using limits = std::numeric_limits<typename float_traits::fp_t>;
  using du_int = unsigned long long;

  if (a == 0)
    return 0;
  const unsigned N = sizeof(du_int) * __CHAR_BIT__;
  const du_int s = std::is_same<ll_or_ull, long long>::value
    ? du_int(0) : a >> (N-1);
  a = (a ^ s) - s;
  int sd = N - rep_clz(a);  /* number of significant digits */
  int e = sd - 1;             /* exponent */
  if (sd > limits::digits)
  {
    /*  start:  0000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQxxxxxxxxxxxxxxxxxx
     *  finish: 000000000000000000000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQR
     *                                                12345678901234567890123456
     *  1 = msb 1 bit
     *  P = bit limits::digits-1 bits to the right of 1
     * Q = bit limits::digits bits to the right of 1
     *  R = "or" of all bits to the right of Q
    */
    switch (sd)
    {
    case limits::digits + 1:
      a <<= 1;
      break;
    case limits::digits + 2:
        break;
    default:
      a = (du_int(a) >> (sd - (limits::digits+2))) |
          ((a & (du_int(-1) >> ((N + limits::digits+2) - sd))) != 0);
    };
    /* finish: */
    a |= (a & 4) != 0;  /* Or P into R */
    ++a;  /* round - this step may add a significant bit */
    a >>= 2;  /* dump Q and R */
    /* a is now rounded to limits::digits or limits::digits+1 bits */
    if (a & (du_int(1) << limits::digits))
    {
      a >>= 1;
      ++e;
    }
    /* a is now rounded to limits::digits bits */
  }
  else
  {
    a <<= (limits::digits - sd);
    /* a is now rounded to limits::digits bits */
  }

  using rep_t = typename float_traits::rep_t;
  static_assert(
    std::is_same<uint32_t, rep_t>::value ||
    std::is_same<uint64_t, rep_t>::value,
    "uint32_t or uint64_t only");
  if (std::is_same<uint64_t, rep_t>::value) {
    uint32_t high = (uint32_t(s) & 0x80000000) |      /* sign */
                    ((e + 1023) << 20)         |      /* exponent */
                    (uint32_t(a >> 32) & 0x000FFFFF); /* mantissa-high */
    uint32_t low = uint32_t(a);                       /* mantissa-low */
    return uint64_t(high) << 32 | low;
  }
  else {
    return (uint32_t(s) & 0x80000000) | /* sign */
           ((e + 127) << 23)          | /* exponent */
           (uint32_t(a) & 0x007FFFFF);  /* mantissa */
  }
}

template<class Int>
constexpr Int cfloat_pow(Int a, int b)
{
  using traits = float_traits_from<Int>;
  const bool recip = b < 0;
  Int r = cfloat_i2f<traits>(1u);
  while (1)
  {
    if (b & 1)
      r = cfloat_mul(r, a);
    b /= 2;
    if (b == 0)
      break;
    cfloat_mul(a, a);
  }
  return recip ? cfloat_div(cfloat_i2f<traits>(1u), r) : r;
}


template<class Int>
constexpr bool cfloat_isnan(Int a)
{
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  return aAbs > cons::infRep;
}

template<class Int>
constexpr bool cfloat_isinf(Int a)
{
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  return aAbs == cons::infRep;
}

template<class Int>
constexpr bool cfloat_isfinite(Int a)
{
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  return aAbs < cons::infRep;
}

template<class Int>
constexpr bool cfloat_iszero(Int a)
{
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  return !aAbs;
}

template<class Int>
constexpr bool cfloat_issubnormal(Int a)
{
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  return aAbs == cons::oneRep;
}

template<class Int>
constexpr bool cfloat_isnormal(Int a)
{
  return cfloat_isfinite(a) && !cfloat_iszero(a) && !cfloat_issubnormal(a);
}

template<class Int>
constexpr int cfloat_fpclassify(Int a)
{
  if (cfloat_isnan(a)) return FP_NAN;
  if (cfloat_isinf(a)) return FP_INFINITE;
  if (cfloat_iszero(a)) return FP_ZERO;
  if (cfloat_issubnormal(a)) return FP_SUBNORMAL;
  return FP_NORMAL;
}

template<class Int>
constexpr bool cfloat_unordered(Int a, Int b)
{
  //return cfloat_isnan(a) || cfloat_isnan(b);
  using cons = float_traits_from<Int>;
  Int const aAbs = a & cons::absMask;
  Int const bAbs = b & cons::absMask;
  return (aAbs | bAbs) > cons::infRep;
}

template<class Int>
constexpr bool cfloat_signbit(Int a)
{
  using cons = float_traits_from<Int>;
  return bool(a & cons::signBit);
}

enum class le_result
{
  less      = -1,
  equal     =  0,
  greater   =  1,
  unordered =  2
};

template<class Int>
constexpr le_result cfloat_compare(Int a, Int b)
{
  using cons = float_traits_from<Int>;
  using rep_t = typename cons::rep_t;
  using srep_t = typename cons::srep_t;

  const srep_t aInt = a;
  const srep_t bInt = b;
  const rep_t aAbs = aInt & cons::absMask;
  const rep_t bAbs = bInt & cons::absMask;

  // If either a or b is NaN, they are unordered.
  if (aAbs > cons::infRep || bAbs > cons::infRep) return le_result::unordered;

  // If a and b are both zeros, they are equal.
  if ((aAbs | bAbs) == 0) return le_result::equal;

  // If at least one of a and b is positive, we get the same result comparing
  // a and b as signed integers as we would with a fp_ting-point compare.
  if ((aInt & bInt) >= 0) {
    if (aInt < bInt) return le_result::less;
    else if (aInt == bInt) return le_result::equal;
    else return le_result::greater;
  }

  // Otherwise, both are negative, so we need to flip the sense of the
  // comparison to get the correct result.  (This assumes a twos- or ones-
  // complement integer representation; if integers are represented in a
  // sign-magnitude representation, then this flip is incorrect).
  else {
    if (aInt > bInt) return le_result::less;
    else if (aInt == bInt) return le_result::equal;
    else return le_result::greater;
  }
}

template<class Int>
constexpr bool cfloat_isgreater(Int a, Int b)
{
  return cfloat_compare(a, b) == le_result::greater;
}

template<class Int>
constexpr bool cfloat_isless(Int a, Int b)
{
  return cfloat_compare(a, b) == le_result::less;
}

template<class Int>
constexpr bool cfloat_isgreaterequal(Int a, Int b)
{
  return unsigned(cfloat_compare(a, b)) <= unsigned(le_result::greater);
}

template<class Int>
constexpr bool cfloat_islessequal(Int a, Int b)
{
  return int(cfloat_compare(a, b)) <= int(le_result::equal);
}

template<class Int>
constexpr bool cfloat_islessgreater(Int a, Int b)
{
  return unsigned(cfloat_compare(a, b)) & 1u;
}

} } }
