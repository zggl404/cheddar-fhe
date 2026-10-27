// Coefficient data from ACE util/src/app_composite_poly.cxx.
// Copyright (c) Ant Group Co., Ltd.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include <array>
#include <vector>
namespace cheddar::resnet {
inline const std::array<std::vector<double>,3>& SignPolynomials() {
  static const std::array<std::vector<double>,3> p{{
{0.,
             1.274244441439567055e+00,
             0.,
             -4.274610154279958607e-01,
             0.,
             2.598417608934820988e-01,
             0.,
             -1.894160321998888952e-01,
             0.,
             1.516157904980795224e-01,
             0.,
             -1.289471808964555988e-01,
             0.,
             1.148389592593351827e-01,
             0.,
             -1.006755030034787834e+00,},
{0.,
             1.504797731281392936e+00,
             0.,
             -1.262993831946355172e-01,
             0.,
             5.310374803122150933e-01,
             0.,
             -4.763164287058726520e-01,
             0.,
             1.404090303951424090e-01,
             0.,
             -1.856485351687612792e-02,
             0.,
             1.142402954164560992e-03,
             0.,
             -2.667926441648920576e-05,},
{0., 1.258870573407572691e+00, 0., -3.830661449095234539e-01,
             0., 1.909371044429533648e-01, 0., -1.025700865042690896e-01,
             0., 5.364833181833868897e-02, 0., -2.602904444646918572e-02,
             0., 1.119529495100999271e-02, 0., -3.976394146723259693e-03,
             0., 1.080475747158062428e-03, 0., -2.115428631766840754e-04,
             0., 2.840163212584644305e-05, 0., -2.461531419370990484e-06,
             0., 1.235599278444410819e-07, 0., -2.723078631019510824e-09,}
  }};
  return p;
}
inline double Chebyshev(const std::vector<double>& p, double x) {
  double b1=0,b2=0;
  for(int i=int(p.size())-1;i>0;--i) {
    double next=2*x*b1-b2+p[i]; b2=b1; b1=next;
  }
  return x*b1-b2+p[0];
}

// Express p(a*x)/b in the Chebyshev basis of x. Keep intermediate arithmetic
// wider than double: evaluating the original T_27 at inputs near 1.56 is badly
// conditioned. This changes coordinates, not the composed approximation.
inline std::vector<double> RescaleChebyshev(const std::vector<double>& p,
                                           long double a, long double b) {
  const size_t n=p.size();
  std::vector<long double> previous(n),current(n),result(n);
  previous[0]=1;
  result[0]=p[0];
  current[1]=a;
  for(size_t k=1;k<n;++k) {
    for(size_t j=0;j<n;++j) result[j]+=static_cast<long double>(p[k])*current[j];
    if(k+1==n) break;
    std::vector<long double> next(n);
    next[1]+=2*a*current[0];
    for(size_t j=1;j<n;++j) {
      next[j-1]+=a*current[j];
      if(j+1<n) next[j+1]+=a*current[j];
    }
    for(size_t j=0;j<n;++j) next[j]-=previous[j];
    previous=std::move(current);current=std::move(next);
  }
  std::vector<double> out(n);
  for(size_t j=0;j<n;++j) out[j]=static_cast<double>(result[j]/b);
  return out;
}
inline const std::array<std::vector<double>,3>& StableSignPolynomials() {
  // z1=p1(x)/2; z2=p2(2*z1)/(7/4); z3=p3((7/4)*z2)/2.
  // Keep both Boot inputs inside [-1,1]. The 7/4 coordinate lowers
  // cancellation in the final Chebyshev evaluation without changing composition.
  static const std::array<std::vector<double>,3> p{{
    RescaleChebyshev(SignPolynomials()[0],1,2),
    RescaleChebyshev(SignPolynomials()[1],2,1.75L),
    RescaleChebyshev(SignPolynomials()[2],1.75L,2)}};
  return p;
}
inline double PolyReLU(double x, double range=10.0) {
  double y=x/range;
  for(const auto& p:StableSignPolynomials()) y=Chebyshev(p,y);
  return x*(0.5+y);
}
}  // namespace cheddar::resnet
