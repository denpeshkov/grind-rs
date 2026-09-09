use std::fmt::Debug;
use std::hash::{self, Hash};
use std::marker::PhantomData;
use std::ops;

pub trait Monoid<T>
where
    T: Clone,
{
    fn op(a: &T, b: &T) -> T;
    fn id() -> T;
}

/// A segment tree.
/// Allows range queries and single element updates.
#[derive(Debug, Clone)]
pub struct SegmentTree<T, M>
where
    T: Clone,
    M: Monoid<T>,
{
    tree: Vec<T>,
    _m: PhantomData<M>,
}

impl<T, M> Eq for SegmentTree<T, M>
where
    T: Clone + Eq,
    M: Monoid<T> + Eq,
{
}

impl<T, M> PartialEq for SegmentTree<T, M>
where
    T: Clone + PartialEq,
    M: Monoid<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.tree[self.len()..] == other.tree[other.len()..]
    }
}

impl<T, M> Hash for SegmentTree<T, M>
where
    T: Clone + Hash,
    M: Monoid<T>,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.tree[self.len()..].hash(state);
    }
}

impl<T, M> SegmentTree<T, M>
where
    T: Clone,
    M: Monoid<T>,
{
    /// Creates a new [`SegmentTree<O>`] of size `size`.
    pub fn new(size: usize) -> Self {
        Self {
            tree: vec![M::id(); 2 * size],
            _m: PhantomData,
        }
    }

    /// Set's the element at `index` to the `value`.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    pub fn set(&mut self, index: usize, value: T) {
        let mut i = index + (self.tree.len() >> 1);
        self.tree[i] = value;
        while i > 1 {
            i >>= 1;
            self.tree[i] = M::op(&self.tree[i << 1], &self.tree[(i << 1) + 1]);
        }
    }

    /// Provides a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.tree.get(index + (self.tree.len() >> 1))
    }

    /// Computes the query on the specified range.
    ///
    /// # Panics
    /// This function panics if:
    /// - `start > end`.
    /// - Bounded on either end and past the `len` of the tree.
    pub fn query<R>(&self, range: R) -> T
    where
        R: ops::RangeBounds<usize>,
        T: Debug,
    {
        let (mut start, mut end) = self.range_start_end(range);
        let mut result = M::id();
        while start < end {
            if start & 1 != 0 {
                // Start is odd - it's a right child.
                result = M::op(&result, &self.tree[start]);
                start += 1;
            }
            if end & 1 != 0 {
                // End+1 is even (we are using exclusive range) - it's a left child.
                end -= 1;
                result = M::op(&result, &self.tree[end]);
            }
            start >>= 1;
            end >>= 1;
        }
        result
    }

    /// Returns the number of elements in the tree.
    pub fn len(&self) -> usize {
        self.tree.len() >> 1
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[track_caller]
    fn range_start_end<R>(&self, range: R) -> (usize, usize)
    where
        R: ops::RangeBounds<usize>,
    {
        use ops::Bound::*;

        let n = self.len();
        let start = match range.start_bound() {
            Included(&i) => i,
            Excluded(&i) => i + 1,
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(&i) => i + 1,
            Excluded(&i) => i,
            Unbounded => n,
        };

        assert!(
            start <= end,
            "range start is greater than range end in SegmentTree"
        );
        assert!(end <= n, "range end is greater than SegmentTree length");
        (n + start, n + end)
    }
}

impl<T, M> ops::Index<usize> for SegmentTree<T, M>
where
    T: Clone,
    M: Monoid<T>,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        let n = self.tree.len() >> 1;
        &self.tree[n + index]
    }
}

impl<T, M> FromIterator<T> for SegmentTree<T, M>
where
    T: Clone,
    M: Monoid<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let leaves: Vec<_> = iter.into_iter().collect();
        let n = leaves.len();

        let mut tree = Vec::with_capacity(2 * n);
        tree.resize(n, M::id());
        tree.extend(leaves);
        for i in (1..n).rev() {
            tree[i] = M::op(&tree[2 * i], &tree[2 * i + 1])
        }
        Self {
            tree,
            _m: PhantomData,
        }
    }
}

impl<T, M, const N: usize> From<[T; N]> for SegmentTree<T, M>
where
    T: Clone,
    M: Monoid<T>,
{
    fn from(arr: [T; N]) -> Self {
        let mut tree = Vec::with_capacity(2 * N);
        tree.resize(N, M::id());
        tree.extend(arr);
        for i in (1..N).rev() {
            tree[i] = M::op(&tree[2 * i], &tree[2 * i + 1])
        }
        Self {
            tree,
            _m: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::array::uniform10;
    use proptest::collection::*;
    use proptest::prelude::*;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Add;
    impl Monoid<i32> for Add {
        fn op(a: &i32, b: &i32) -> i32 {
            a + b
        }
        fn id() -> i32 {
            0
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Max;
    impl Monoid<i32> for Max {
        fn op(a: &i32, b: &i32) -> i32 {
            *a.max(b)
        }
        fn id() -> i32 {
            i32::MIN
        }
    }

    #[test]
    fn test_new() {
        let mut tree: SegmentTree<i32, Add> = SegmentTree::new(42);
        assert_eq!(tree.get(0), Some(&Add::id()));
        assert_eq!(tree.get(41), Some(&Add::id()));

        assert_eq!(tree.query(11..19), Add::id());
        tree.set(13, 42);
        assert_eq!(tree.query(11..19), 42);

        let mut tree: SegmentTree<i32, Max> = SegmentTree::new(42);
        assert_eq!(tree.get(0), Some(&Max::id()));
        assert_eq!(tree.get(41), Some(&Max::id()));

        assert_eq!(tree.query(11..19), Max::id());
        tree.set(13, 42);
        assert_eq!(tree.query(11..19), 42);
    }

    #[test]
    #[should_panic(expected = "range start is greater than range end in SegmentTree")]
    fn test_query_panic_1() {
        let tree: SegmentTree<i32, Add> = SegmentTree::from([1, 2, 3, 4, 5]);
        #[allow(clippy::reversed_empty_ranges)]
        tree.query(3..1);
    }

    #[test]
    #[should_panic(expected = "range end is greater than SegmentTree length")]
    fn test_query_panic_2() {
        let tree: SegmentTree<i32, Add> = SegmentTree::from([1, 2, 3, 4, 5]);
        tree.query(1..6);
    }

    #[test]
    #[should_panic]
    fn test_set_panic() {
        let mut tree: SegmentTree<i32, Add> = SegmentTree::from([1, 2, 3, 4, 5]);
        tree.set(6, 42);
    }

    #[test]
    #[should_panic]
    fn test_index_panic() {
        let tree: SegmentTree<i32, Add> = SegmentTree::from([1, 2, 3, 4, 5]);
        _ = tree[6];
    }

    proptest! {
        #[test]
        fn prop_from_iter(v in vec(-1024..=1024, 0..=64)) {
             let tree: SegmentTree<i32, Add> = v.iter().cloned().collect();
             prop_assert_eq!(tree.len(), v.len());
             prop_assert_eq!(tree.is_empty(), v.is_empty());
        }

        #[test]
        fn prop_from_array(arr in uniform10(-1024..=1024)) {
            let tree = SegmentTree::<i32, Add>::from(arr);
            prop_assert_eq!(tree.len(), arr.len());
            prop_assert_eq!(tree.is_empty(), arr.is_empty());
        }
    }

    proptest! {
        #[test]
        fn prop_eq(v1 in vec(-8..=8, 0..=8), v2 in vec(-8..=8, 0..=8)) {
            let tree1: SegmentTree<i32, Add> = v1.iter().cloned().collect();
            let tree2: SegmentTree<i32, Add> = v2.iter().cloned().collect();
            prop_assert_eq!(tree1.eq(&tree2), v1.eq(&v2));
        }
    }

    proptest! {
        #[test]
        fn prop_hash(v in vec(-1024..=1024, 0..=64)) {
            use hash::Hasher;

            let tree_add: SegmentTree<i32, Add> = v.iter().cloned().collect();
            let tree_max: SegmentTree<i32, Max> = v.iter().cloned().collect();

            let mut hasher = hash::DefaultHasher::new();
            tree_add.hash(&mut hasher);
            let tree_add_hash = hasher.finish();

            let mut hasher = hash::DefaultHasher::new();
            tree_max.hash(&mut hasher);
            let tree_max_hash = hasher.finish();

            prop_assert_eq!(tree_add_hash, tree_max_hash);
        }
    }

    /// Generates all combinations of range bounds for the given tree length.
    fn range_bounds(
        tree_len: usize,
    ) -> impl Strategy<Value = (ops::Bound<usize>, ops::Bound<usize>)> {
        (0..tree_len, 0..tree_len)
            // Generate an ordered pair (start <= end).
            .prop_map(|(a, b)| (a.min(b), a.max(b)))
            // Generate bound types combinations.
            .prop_flat_map(|(start, end)| {
                let start = prop_oneof![
                    Just(ops::Bound::Included(start)),
                    Just(ops::Bound::Excluded(start)),
                    Just(ops::Bound::Unbounded),
                ];
                let end = prop_oneof![
                    Just(ops::Bound::Included(end)),
                    Just(ops::Bound::Excluded(end)),
                    Just(ops::Bound::Unbounded),
                ];
                (start, end)
            })
        .prop_filter(
            "Excluded(x)..Excluded(x) is an invalid range",
            |bounds| !matches!(bounds, (ops::Bound::Excluded(s), ops::Bound::Excluded(e)) if s == e),
        )
    }

    prop_compose! {
        /// Generates a strategy returning a vector and range bounds.
        fn vec_and_range()(v in vec(-1024..=1024, 1..=128))
                        (range in range_bounds(v.len()), v in Just(v))
                        -> (Vec<i32>, (ops::Bound<usize>, ops::Bound<usize>)) {
            (v, range)
       }
    }

    #[derive(Copy, Clone, Debug)]
    enum Ops<T, R> {
        Get(usize),
        Set(usize, T),
        Query(R),
    }

    /// Generates [SegmentTree] operations.
    fn ops(tree_len: usize) -> impl Strategy<Value = Ops<i32, ops::RangeInclusive<usize>>> {
        prop_oneof![
            (0..tree_len).prop_map(Ops::Get),
            (0..tree_len, -128..=128).prop_map(|(index, value)| Ops::Set(index, value)),
            (0..tree_len, 0..tree_len)
                .prop_map(|(a, b)| (a.min(b), a.max(b))) // Generate an ordered pair (start <= end).
                .prop_map(|(start, end)| Ops::Query(start..=end)),
        ]
    }

    prop_compose! {
        /// Generates a strategy returning a vector and operations for SegmentTree based on it.
        fn vec_and_ops()(v in vec(-1024..=1024, 1..=64))
                        (ops in vec(ops(v.len()), 1..=512), v in Just(v))
                        -> (Vec<i32>, Vec<Ops<i32, ops::RangeInclusive<usize>>>) {
            (v, ops)
       }
    }

    proptest! {
        #[test]
        fn prop_operations((mut v, ops) in vec_and_ops()) {
            let mut tree: SegmentTree<i32, Add> = v.iter().cloned().collect();
            for op in ops {
                match op {
                    Ops::Get(index) => {
                        prop_assert_eq!(tree.get(index), v.get(index));
                        prop_assert_eq!(tree[index],v[index]);
                    }
                    Ops::Set(index, value) => {
                        tree.set(index, value);
                        v[index] = value;
                    },
                    Ops::Query(range) => {
                        prop_assert_eq!(tree.query(range.clone()), v[range.clone()].iter().sum())
                    },
                }
            }
        }
    }

    proptest! {
        #[test]
        fn prop_query((v, range) in vec_and_range()) {
            let tree: SegmentTree<i32, Add> = v.iter().cloned().collect();
            prop_assert_eq!(tree.query(range), v[range].iter().sum())
        }
    }
}
