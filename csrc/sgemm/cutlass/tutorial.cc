#include "cute/algorithm/functional.hpp"
#include "cute/layout.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print_latex.hpp"
#include <iostream>

using namespace cute;

template <class Shape, class Stride>
void print1D(Layout<Shape, Stride> const& layout) {
    for (int i = 0; i < size(layout); ++i) {
        printf("%3d  ", layout(i));
    }
}

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const& layout) {
    for (int m = 0; m < size<0>(layout); ++m) {
        for (int n = 0; n < size<1>(layout); ++n) {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    Layout s8 = make_layout(Int<8>{});
    Layout d8 = make_layout(8);
    Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{}, 4));
    Layout s2xd4_a =   make_layout(make_shape (Int< 2>{}, 4), make_stride(Int<12>{}, Int<1>{}));
    Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});
    Layout s2xh4 =     make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1)));
    Layout s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});

    // coalesce
    // Layout a = Layout<Shape<_2,Shape<_1,_6>>, Stride<_1,Stride<_6,_2>>>{};
    // Layout b = coalesce(a, Step<_1, Step<_1, _1>>{});
    // print(b);

    // composition
    // Layout a = Layout<Shape<_10, _2>, Stride<_16, _4>>{};
    // Layout b = Layout<Shape<_5,_4>, Stride<_1,_5>>{};
    // print_latex(composition(a, b));
    
    // complement
    // Layout a = complement(
    //     make_layout(make_shape(Int<4>{}, Int<6>{}), make_stride(Int<1>{}, Int<4>{})),
    //     make_shape(Int<6>{}, Int<8>{})
    // );
    // print(a);

    // division
    // Layout L = make_layout(make_shape(Int<4>{}, Int<2>{}, Int<3>{}), make_stride(Int<2>{}, Int<1>{}, Int<8>{}));
    // Layout tiler = make_layout(Int<4>{}, Int<2>{});
    // Layout comp = complement(tiler, size(L));
    // print_latex(make_layout(tiler, comp));
    // print(composition(L, make_layout(tiler, comp)));

    auto L = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})), 
                         make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
    auto tiler = make_tile(Layout<_3,_3>{},
                           Layout<Shape <_2,_4>,
                           Stride<_1,_8>>{});
    auto ld = logical_divide(L, tiler);                      
    auto zd = zipped_divide(L, tiler);
    auto td = tiled_divide(L, tiler);
    print(ld);
    std::cout << "\n";
    print(zd);
    std::cout << "\n";
    print(td);

    std::cout << std::endl;
    return 0;
}

/*
notes:
- majorness is simply which mode has stride-1
- a stride of (1, 2) is +1 down the rows (increasing dim-0), +2 across the columns (increasing dim-1)
- any layout can take a 1-dimensional coordinate (ex. layout(i))
  - thus any layout can be viewed as a vector (ex. `test_vec`)
*/
