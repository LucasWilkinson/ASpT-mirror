#ifndef KEY_VALUE_QSORT_
#define KEY_VALUE_QSORT_

template <typename sKey>
bool greaterThanFunction(const sKey &a, const sKey &b) {
  return a > b;
}

template <typename sKey>
bool lessThanFunction(const sKey &a, const sKey &b) {
  return a < b;
}

template <typename sKey, typename sQValue>
void key_value_qsort (sKey *keys, sQValue *values, long n,
    bool (*lessThan)(const sKey &a, const sKey &b)) {
  if (n < 2)
    return;
  int p = keys[n >> 1];
  sKey *l = keys;
  sKey *r = keys + n - 1;
  while (l <= r) {
    if (lessThan(*l, p)) {
      l++;
    } else if (lessThan(p, *r)) {
      r--;
    } else {
      sQValue *vl = values + (l - keys);
      sQValue *vr = values + (r - keys);
      sQValue vt = *vl; *vl = *vr; *vr = vt;
      int t = *l; *l++ = *r; *r-- = t;
    }
  }
  key_value_qsort<sKey, sQValue>(keys, values, r - keys + 1, lessThan);
  key_value_qsort<sKey, sQValue>(l, values + (l - keys), keys + n - l, lessThan);
}

template <typename sKey, typename sQValue>
void key_value_qsort(sKey *keys, sQValue *values, long n) {
  bool (*functionPointer)(const sKey &, const sKey &) = &(lessThanFunction<sKey>);
  key_value_qsort<sKey, sQValue>(keys, values, n, functionPointer);
}
#endif
