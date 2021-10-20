//UnrollerP: loops over given size, partial unrolled
template<int InnerUnroll = 8, int Begin = 0>
public:
struct UnrollerP {
    template<typename Lambda>
    static void step(size_t N, Lambda& func) {
        size_t i = Begin;
        for (; i < N - InnerUnroll; i += InnerUnroll) {
            UnrollerInternal<>::step(func, i);
        }
        for (; i < N; ++i) {
            func(i);
        }
        
    }
private:
    //start of UnrollerInternal
    template<size_t Offset = 0>
    struct UnrollerInternal {
        template<typename Lambda>
        static void step(Lambda& func, size_t i) {
            func(i + Offset);
            UnrollerInternal<Offset + 1>::step(func, i);
        }
    };

    //end of UnrollerInternal
    template<>
    struct UnrollerInternal<InnerUnroll> {
        template<typename Lambda>
        static void step(Lambda& func, size_t i) {
        }
    };

};

// Usage:
// int numbers; //get 'numbers' at runtime
// int *arr = new int[numbers];
// int sum = 0, tmp;

// unroll the loop 8 times, offset is 0 so 
// the range is from 0 to numbers
// UnrollerP<8>::step(numbers, [&] (size_t i) {
//    arr[i] = i;
//    tmp = arr[i] + sum;
//    arr[i] = sum;
//    sum = tmp;
// }
// );

