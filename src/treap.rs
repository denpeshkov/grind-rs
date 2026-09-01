use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{self, Hash};
use std::iter::FusedIterator;
use std::ops::{self, Bound, Index, RangeBounds};

/// An ordered map based on a treap (cartesian tree).
///
/// It is a logic error for a key to be modified in such a way that the key’s
/// ordering relative to any other key, as determined by the [`Ord`] trait,
/// changes while it is in the map.
#[derive(Debug, Clone)]
pub struct Treap<K, V> {
    root: Option<Box<Node<K, V>>>,
}

impl<K, Q, V> Index<&Q> for Treap<K, V>
where
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    type Output = V;
    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V> PartialEq for Treap<K, V>
where
    K: Ord,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<K, V> Eq for Treap<K, V>
where
    K: Ord,
    V: Eq,
{
}

impl<K, V> PartialOrd for Treap<K, V>
where
    K: Ord,
    V: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V> Ord for Treap<K, V>
where
    K: Ord,
    V: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V> Hash for Treap<K, V>
where
    K: Hash,
    V: Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.len().hash(state);
        for elt in self {
            elt.hash(state);
        }
    }
}

impl<K, V> Treap<K, V> {
    /// Makes a new, empty [`Treap`].
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Clears the map, removing all elements.
    pub fn clear(&mut self) {
        self.root = None
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.root.as_deref().map_or(0, |n| n.size)
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Ord, V> Treap<K, V> {
    /// Returns a reference to the value corresponding to the key.
    /// The ordering on the borrowed form **must** match the ordering on the key type.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.root.as_deref()?.get(key).map(|n| &n.value)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    /// The ordering on the borrowed form must match the ordering on the key type.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.root.as_deref_mut()?.get_mut(key).map(|n| &mut n.value)
    }

    /// Inserts a key-value pair into the map.
    /// If the map did not have this key present, [`None`] is returned.
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let Some(root) = self.root.take() else {
            self.root = Some(Box::new(Node::new(key, value)));
            return None;
        };
        let (l, x, r) = root.split(&key);
        // Update x in place if it already exists, or create a fresh x.
        let (old, x) = match x {
            Some(mut x) => (Some(std::mem::replace(&mut x.value, value)), x),
            None => (None, Box::new(Node::new(key, value))),
        };
        self.root = Node::merge(l, Node::merge(Some(x), r));
        old
    }

    /// Removes a key from the map, returning the stored key and value if the key was previously in
    /// the map.
    /// The ordering on the borrowed form must match the ordering on the key type.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (l, x, r) = self.root.take()?.split(key);
        self.root = Node::merge(l, r);
        x.map(|x| (x.key, x.value))
    }

    /// Returns a key-value pair at the given rank.
    pub fn nth(&self, rank: usize) -> Option<(&K, &V)> {
        self.root.as_deref()?.nth(rank).map(|n| (&n.key, &n.value))
    }

    /// Removes and returns a key-value pair at the given rank.
    pub fn remove_nth(&mut self, rank: usize) -> Option<(K, V)> {
        let root = self.root.take()?;
        let (root, removed) = root.remove_nth(rank);
        self.root = root;
        removed.map(|node| (node.key, node.value))
    }

    /// Returns a rank of the key: the number of keys `<` key.
    /// A key doesn't need to exist in the map.
    pub fn rank<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.root.as_deref().map_or(0, |n| n.rank(key))
    }

    /// Constructs an iterator over a sub-range of elements in the map.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are [Excluded](ops::Bound::Excluded).
    pub fn range<Q, R>(&self, range: R) -> Range<'_, K, V, R>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: ops::RangeBounds<Q>,
    {
        assert_range(&range);
        Range {
            node: self.root.as_deref(),
            stack: vec![],
            range,
        }
    }

    /// Constructs a mutable iterator over a sub-range of elements in the map.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are [Excluded](ops::Bound::Excluded).
    pub fn range_mut<Q, R>(&mut self, range: R) -> RangeMut<'_, K, V, R>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: ops::RangeBounds<Q>,
    {
        assert_range(&range);
        RangeMut {
            node: self.root.as_deref_mut(),
            stack: vec![],
            range,
        }
    }

    /// Gets an iterator over the entries of the map, sorted by key.
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.into_iter()
    }

    /// Gets a mutable iterator over the entries of the map, sorted by key.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.into_iter()
    }
}

/// # Panics
/// This function panics if:
/// - `start > end`.
/// - `start == end` and both are `Excluded`.
fn assert_range<T, R>(range: &R)
where
    T: Ord + ?Sized,
    R: RangeBounds<T>,
{
    match (range.start_bound(), range.end_bound()) {
        (Bound::Excluded(s), Bound::Excluded(e)) if s == e => {
            panic!("range start and end are equal and excluded in Treap")
        }
        (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e))
            if s > e =>
        {
            panic!("range start is greater than range end in Treap")
        }
        _ => {}
    }
}

impl<K, V> Default for Treap<K, V> {
    /// Creates an empty [`Treap`].
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, K, V> IntoIterator for &'a Treap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            node: self.root.as_deref(),
            stack: vec![],
            len: self.len(),
        }
    }
}

/// An iterator over the entries of a [`Treap`].
pub struct Iter<'a, K, V> {
    node: Option<&'a Node<K, V>>,
    stack: Vec<&'a Node<K, V>>,
    len: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let node = loop {
            match self.node.take() {
                Some(node) => {
                    self.stack.push(node);
                    self.node = node.left.as_deref();
                    // Continue the loop.
                }
                None => {
                    let node = self.stack.pop()?;
                    self.node = node.right.as_deref();
                    break node;
                }
            }
        };
        Some((&node.key, &node.value))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

impl<'a, K, V> IntoIterator for &'a mut Treap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        IterMut {
            node: self.root.as_deref_mut(),
            stack: vec![],
            len,
        }
    }
}

/// A mutable iterator over the entries of a [`Treap`].
/// Only values are exclusive borrowed, keys are shared borrowed.
pub struct IterMut<'a, K, V> {
    node: Option<&'a mut Node<K, V>>,

    #[allow(clippy::type_complexity)]
    /// Holds the pieces of the node we need to visit later: (key, value, right child)
    stack: Vec<(&'a K, &'a mut V, Option<&'a mut Node<K, V>>)>,
    len: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        let kv = loop {
            match self.node.take() {
                Some(node) => {
                    let left = node.left.as_deref_mut();
                    self.node = left;
                    // Split the &'a mut Node into separate pieces for disjoint field borrowing.
                    self.stack
                        .push((&node.key, &mut node.value, node.right.as_deref_mut()));
                    // Continue the loop.
                }
                None => {
                    let (key, value, right_node) = self.stack.pop()?;
                    self.node = right_node;
                    break (key, value);
                }
            }
        };
        Some(kv)
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}

/// An owning iterator over the entries of a BTreeMap, sorted by key.
impl<K, V> IntoIterator for Treap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        IntoIter {
            node: self.root,
            stack: vec![],
            len,
        }
    }
}

/// An iterator over the entries of a [`Treap`].
pub struct IntoIter<K, V> {
    node: Option<Box<Node<K, V>>>,
    stack: Vec<Box<Node<K, V>>>,
    len: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        let node = loop {
            match self.node.take() {
                Some(mut node) => {
                    let left = node.left.take();
                    self.stack.push(node);
                    self.node = left;
                    // Continue the loop.
                }
                None => {
                    let mut node = self.stack.pop()?;
                    self.node = node.right.take();
                    break node;
                }
            }
        };
        Some((node.key, node.value))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}

impl<K, V> FromIterator<(K, V)> for Treap<K, V>
where
    K: Ord,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Treap::new();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl<K, V> Extend<(K, V)> for Treap<K, V>
where
    K: Ord,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K, V> Extend<(&'a K, &'a V)> for Treap<K, V>
where
    K: Ord + Copy,
    V: Copy,
{
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(*key, *value);
        }
    }
}

/// An iterator over a sub-range of entries in a [Treap]
pub struct Range<'a, K, V, R> {
    node: Option<&'a Node<K, V>>,
    stack: Vec<&'a Node<K, V>>,
    range: R,
}

impl<'a, K, V, R> Iterator for Range<'a, K, V, R>
where
    K: Ord,
    R: ops::RangeBounds<K>,
{
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let node = loop {
            match self.node.take() {
                Some(node) => {
                    // Test if key is too small.
                    let is_too_small = match self.range.start_bound() {
                        Bound::Included(key) => &node.key < key,
                        Bound::Excluded(key) => &node.key <= key,
                        Bound::Unbounded => false,
                    };
                    if is_too_small {
                        self.node = node.right.as_deref();
                    } else {
                        // Node might be in range.
                        self.stack.push(node);
                        self.node = node.left.as_deref();
                    }
                }
                None => {
                    let node = self.stack.pop()?;
                    // Nodes are only pushed after passing `too_small`, so only
                    // the upper bound needs to be checked here.
                    if !self.range.contains(&node.key) {
                        // All remaining nodes are also too large, so we are finished.
                        self.stack.clear();
                        return None;
                    }
                    self.node = node.right.as_deref();
                    break node;
                }
            }
        };
        Some((&node.key, &node.value))
    }
}

impl<'a, K, V, R> FusedIterator for Range<'a, K, V, R>
where
    K: Ord,
    R: ops::RangeBounds<K>,
{
}

/// A mutable iterator over a sub-range of entries in a [Treap].
pub struct RangeMut<'a, K, V, R> {
    node: Option<&'a mut Node<K, V>>,
    #[allow(clippy::type_complexity)]
    /// Holds the pieces of the node we need to visit later: (key, value, right child)
    stack: Vec<(&'a K, &'a mut V, Option<&'a mut Node<K, V>>)>,
    range: R,
}

impl<'a, K, V, R> Iterator for RangeMut<'a, K, V, R>
where
    K: Ord,
    R: ops::RangeBounds<K>,
{
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        let kv = loop {
            match self.node.take() {
                Some(node) => {
                    // Test if key is too small.
                    let is_too_small = match self.range.start_bound() {
                        Bound::Included(key) => &node.key < key,
                        Bound::Excluded(key) => &node.key <= key,
                        Bound::Unbounded => false,
                    };
                    if is_too_small {
                        self.node = node.right.as_deref_mut();
                    } else {
                        // Node might be in range.

                        // Split the &'a mut Node into separate pieces for disjoint fieldborrowing.
                        self.stack
                            .push((&node.key, &mut node.value, node.right.as_deref_mut()));
                        self.node = node.left.as_deref_mut();
                        // Continue the loop.
                    }
                }
                None => {
                    let (key, value, right_node) = self.stack.pop()?;
                    // Nodes are only pushed after passing `too_small`, so only
                    // the upper bound needs to be checked here.
                    if !self.range.contains(key) {
                        // All remaining nodes are also too large, so we are finished.
                        self.stack.clear();
                        return None;
                    }
                    self.node = right_node;
                    break (key, value);
                }
            }
        };
        Some(kv)
    }
}

impl<'a, K, V, R> FusedIterator for RangeMut<'a, K, V, R>
where
    K: Ord,
    R: ops::RangeBounds<K>,
{
}

#[derive(Debug, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
    priority: u64,
    size: usize,
    left: Option<Box<Node<K, V>>>,
    right: Option<Box<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            priority: rand::random(),
            size: 1,
            left: None,
            right: None,
        }
    }

    fn update_size(&mut self) {
        self.size = 1
            + self.left.as_deref().map_or(0, |n| n.size)
            + self.right.as_deref().map_or(0, |n| n.size);
    }

    fn nth(&self, rank: usize) -> Option<&Self> {
        let left_size = self.left.as_deref().map_or(0, |l| l.size);
        match left_size.cmp(&rank) {
            Ordering::Less => self.right.as_deref()?.nth(rank - left_size - 1),
            Ordering::Equal => Some(self),
            Ordering::Greater => self
                .left
                .as_deref()
                .expect("left node should exist") // We checked the size of left-subtree above.
                .nth(rank),
        }
    }

    /// Combines two subtrees into one.
    /// It requires that every key in `left` is less than every key in `right`.
    fn merge(left: Option<Box<Self>>, right: Option<Box<Self>>) -> Option<Box<Self>> {
        match (left, right) {
            (None, None) => None,
            (None, Some(right)) => Some(right),
            (Some(left), None) => Some(left),
            (Some(mut left), Some(mut right)) => {
                // The node with the higher priority becomes the root.
                if left.priority >= right.priority {
                    left.right = Self::merge(left.right, Some(right));
                    left.update_size();
                    Some(left)
                } else {
                    right.left = Self::merge(Some(left), right.left);
                    right.update_size();
                    Some(right)
                }
            }
        }
    }
}

impl<K: Ord, V> Node<K, V> {
    fn get<Q>(&self, key: &Q) -> Option<&Self>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.key.borrow().cmp(key) {
            Ordering::Less => self.right.as_deref().and_then(|r| r.get(key)),
            Ordering::Equal => Some(self),
            Ordering::Greater => self.left.as_deref().and_then(|l| l.get(key)),
        }
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut Self>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.key.borrow().cmp(key) {
            Ordering::Less => self.right.as_deref_mut().and_then(|r| r.get_mut(key)),
            Ordering::Equal => Some(self),
            Ordering::Greater => self.left.as_deref_mut().and_then(|l| l.get_mut(key)),
        }
    }

    fn rank<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.key.borrow().cmp(key) {
            Ordering::Less => {
                1 + self.left.as_deref().map_or(0, |l| l.size)
                    + self.right.as_deref().map_or(0, |r| r.rank(key))
            }
            Ordering::Equal => self.left.as_deref().map_or(0, |l| l.size),
            Ordering::Greater => self.left.as_deref().map_or(0, |l| l.rank(key)),
        }
    }

    #[allow(clippy::type_complexity)]
    /// Partitions subtree rooted at this node into three subtrees:
    ///   1. all keys < key
    ///   2. the node with key, if present
    ///   3. all keys > key
    fn split<Q>(
        mut self: Box<Self>,
        key: &Q,
    ) -> (Option<Box<Self>>, Option<Box<Self>>, Option<Box<Self>>)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.key.borrow().cmp(key) {
            Ordering::Less => {
                let Some(right) = self.right else {
                    return (Some(self), None, None);
                };
                let (rl, x, rr) = right.split(key);
                self.right = rl;
                self.update_size();
                (Some(self), x, rr)
            }
            Ordering::Equal => {
                let (l, r) = (self.left, self.right);
                self.left = None;
                self.right = None;
                self.update_size();
                (l, (Some(self)), r)
            }
            Ordering::Greater => {
                let Some(left) = self.left else {
                    return (None, None, Some(self));
                };
                let (ll, x, lr) = left.split(key);
                self.left = lr;
                self.update_size();
                (ll, x, Some(self))
            }
        }
    }

    fn remove_nth(mut self: Box<Self>, rank: usize) -> (Option<Box<Self>>, Option<Box<Self>>) {
        let left_size = self.left.as_deref().map_or(0, |l| l.size);
        match left_size.cmp(&rank) {
            Ordering::Less => {
                let Some(right) = self.right else {
                    return (None, None);
                };
                let (tight, removed) = Self::remove_nth(right, rank - left_size - 1);
                self.right = tight;
                self.update_size();
                (Some(self), removed)
            }
            Ordering::Equal => {
                let left = self.left.take();
                let right = self.right.take();
                (Self::merge(left, right), Some(self))
            }
            Ordering::Greater => {
                // We already checked that size of left-subtree is greater than rank.
                // This, it must exist.
                let left = self.left.expect("left subtree should exist");
                let (left, removed) = left.remove_nth(rank);
                self.left = left;
                self.update_size();
                (Some(self), removed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::collection::*;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn new_default() {
        assert_eq!(Treap::<i32, i32>::new(), Treap::<i32, i32>::default());
        assert_eq!(Treap::<(), String>::new(), Treap::<(), String>::default());
    }

    #[test]
    fn duplicates() {
        let mut map = Treap::new();
        assert_eq!(map.insert("key1", "val1"), None);
        assert_eq!(map.insert("key1", "val1"), Some("val1"));
        assert_eq!(map.insert("key1", "val2"), Some("val1"));
        assert_eq!(map.insert("key1", "val3"), Some("val2"));
        assert_eq!(map.insert("key2", "val1"), None);
        assert_eq!(map.insert("key1", "val4"), Some("val3"));
        assert_eq!(map.insert("key1", "val4"), Some("val4"));
    }

    #[test]
    fn clear() {
        let mut map = Treap::new();
        assert_eq!(map.insert("key1", "val1"), None);
        assert_eq!(map.insert("key1", "val2"), Some("val1"));
        assert_eq!(map.insert("key2", "val1"), None);
        assert_eq!(map.insert("key2", "val2"), Some("val1"));

        map.clear();
        assert_eq!(map.iter().next(), None);

        assert_eq!(map.insert("key1", "val1"), None);
        assert_eq!(map.insert("key1", "val2"), Some("val1"));
        assert_eq!(map.insert("key2", "val1"), None);
        assert_eq!(map.insert("key2", "val2"), Some("val1"));

        map.clear();
        assert_eq!(map.iter().next(), None);
    }

    #[test]
    fn empty_map_iterators() {
        let mut map = Treap::<i32, i32>::new();
        assert_eq!(map.iter().next(), None);
        assert_eq!(map.iter_mut().next(), None);
        assert_eq!(map.into_iter().next(), None);
    }

    /// Generates valid (start, end) pairs from a key range.
    fn range_bounds_strategy(
        range: ops::RangeInclusive<i32>,
    ) -> impl Strategy<Value = (Bound<i32>, Bound<i32>)> {
        (range.clone(), range)
            // Generate an ordered pair (start <= end).
            .prop_map(|(a, b)| (a.min(b), a.max(b)))
            // Generate bound types combinations.
            .prop_flat_map(|(start, end)| {
                let start = prop_oneof![
                    Just(Bound::Included(start)),
                    Just(Bound::Excluded(start)),
                    Just(Bound::Unbounded),
                ];
                let end = prop_oneof![
                    Just(Bound::Included(end)),
                    Just(Bound::Excluded(end)),
                    Just(Bound::Unbounded),
                ];
                (start, end)
            })
            .prop_filter(
                "Excluded(x)..Excluded(x) is an invalid range",
                |bounds| !matches!(bounds, (Bound::Excluded(s), Bound::Excluded(e)) if s == e),
            )
    }

    proptest! {
        #[test]
        fn prop_iterators(mut std_map in btree_map(any::<i32>(), any::<i32>(), 0..=512)) {
            let mut map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();

            prop_assert!(map.iter().eq(std_map.iter()));

            prop_assert_eq!(map.iter().len(), std_map.iter().len());
            prop_assert!(map.iter_mut().eq(std_map.iter_mut()));
            prop_assert_eq!(map.iter_mut().len(), std_map.iter_mut().len());

            let map_iter = map.into_iter();
            let std_map_iter = std_map.into_iter();
            prop_assert_eq!(&map_iter.len(), &std_map_iter.len());
            prop_assert!(map_iter.eq(std_map_iter));
        }

        #[test]
        fn prop_range(
            mut std_map in btree_map(-32..=32, any::<i32>(), 0..=128),
            range in range_bounds_strategy(-40..=40),
        ) {
            let mut map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();

            prop_assert!(map.range(range).eq(std_map.range(range)));
            prop_assert!(map.range_mut(range).eq(std_map.range_mut(range)));
        }
    }

    #[test]
    #[should_panic(expected = "range start is greater than range end in Treap")]
    fn test_range_panic_1() {
        use ops::Bound::*;

        let mut map = Treap::new();
        map.insert(3, ());
        map.insert(5, ());
        map.insert(8, ());

        map.range((Included(&8), Included(&3)));
    }

    #[test]
    #[should_panic(expected = "range start and end are equal and excluded in Treap")]
    fn test_range_panic_2() {
        use ops::Bound::*;

        let mut map = Treap::new();
        map.insert(3, ());
        map.insert(5, ());
        map.insert(8, ());

        map.range((Excluded(&5), Excluded(&5)));
    }

    #[test]
    #[should_panic(expected = "range start and end are equal and excluded in Treap")]
    fn test_range_panic_3() {
        use ops::Bound::*;

        let mut map = Treap::new();
        map.insert(3, ());
        map.insert(5, ());
        map.insert(8, ());

        map.range((Excluded(&5), Excluded(&5)));
    }

    proptest! {
        #[test]
        fn len(std_map in btree_map(any::<i32>(), any::<i32>(), 0..=64)) {
            let map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();
            prop_assert!(map.len() == std_map.len());
            prop_assert!(map.is_empty() == std_map.is_empty());
        }
    }

    proptest! {
        #[test]
        fn extend(
            initial in btree_map(any::<i32>(), any::<i32>(), 0..=8),
            std_map in btree_map(any::<i32>(), any::<i32>(), 0..=64),
        ) {
            let mut map: Treap<i32, i32> = initial.iter().map(|(k, v)| (*k, *v)).collect();
            let mut map_clone = map.clone();

            map.extend(std_map.iter());
            prop_assert!(std_map.iter().all(|(k, v)| map.get(k).unwrap() == v));

            let mut iter = std_map.into_iter();
            map_clone.extend(&mut iter);
            prop_assert_eq!(map_clone, map);
        }
    }

    #[derive(Copy, Clone, Debug)]
    enum Ops<K, V> {
        Get(K),
        Insert((K, V)),
        Remove(K),
        Clear,
        PopFirst,
        PopLast,
    }

    /// Generates [Treap] operations.
    fn ops_strategy() -> impl Strategy<Value = Ops<i32, i32>> {
        prop_oneof![
            (-64..=64).prop_map(Ops::Get),
            (-64..=64, -64..=64).prop_map(Ops::Insert),
            (-64..=64).prop_map(Ops::Remove),
            Just(Ops::Clear),
            Just(Ops::PopFirst),
            Just(Ops::PopLast),
        ]
    }

    proptest! {
        #[test]
        fn operations(ops in vec(ops_strategy(), 1..=512)) {
            let mut std_map = BTreeMap::new();
            let mut map = Treap::new();

            for op in ops {
                match op {
                    Ops::Get(key) => {
                        prop_assert_eq!(std_map.get(&key), map.get(&key), "test get()");
                        prop_assert_eq!(std_map.get_mut(&key), map.get_mut(&key), "test get_mut()");

                        let want = std::panic::catch_unwind(|| std_map[&key]);
                        let got = std::panic::catch_unwind(|| map[&key]);
                        prop_assert_eq!(want.ok(), got.ok(), "test [] (indexing)");
                    }
                    Ops::Insert((key, value)) => {
                        prop_assert_eq!(std_map.insert(key, value), map.insert(key, value));
                    }
                    Ops::Remove(key) => {
                        prop_assert_eq!(std_map.remove_entry(&key), map.remove(&key));
                    }
                    Ops::Clear => {
                        std_map.clear();
                        map.clear();
                    }
                    Ops::PopFirst => {
                        prop_assert_eq!(std_map.pop_first(), map.remove_nth(0));
                    }
                    Ops::PopLast => {
                        let rank = map.len().saturating_sub(1);
                        prop_assert_eq!(std_map.pop_last(), map.remove_nth(rank));
                    }
                }
            }
            prop_assert!(map.iter().eq(std_map.iter()))
        }
    }

    proptest! {
        #[test]
        fn eq(
            std_map in btree_map(-8..=8, -8..=8, 0..=8),
            std_map2 in btree_map(-8..=8, -8..=8, 0..=8),
        ) {
            let map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();
            let map2: Treap<i32, i32> = std_map2.iter().map(|(k, v)| (*k, *v)).collect();
            prop_assert_eq!(map.eq(&map2), std_map.eq(&std_map2));
        }
    }

    proptest! {
        #[test]
        fn cmp(
            std_map in btree_map(-8..=8, -8..=8, 0..=8),
            std_map2 in btree_map(-8..=8, -8..=8, 0..=8),
        ) {
            let map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();
            let map2: Treap<i32, i32> = std_map2.iter().map(|(k, v)| (*k, *v)).collect();

            prop_assert_eq!(map.partial_cmp(&map2), std_map.partial_cmp(&std_map2));
            prop_assert_eq!(map.cmp(&map2), std_map.cmp(&std_map2));
        }
    }

    proptest! {
        #[test]
        fn hash(std_map in btree_map(-8..=8, -8..=8, 0..=8)) {
            use hash::Hasher;

            let map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();

            let mut hasher = hash::DefaultHasher::new();
            std_map.hash(&mut hasher);
            let want_hash = hasher.finish();

            let mut hasher = hash::DefaultHasher::new();
            map.hash(&mut hasher);
            let got_hash = hasher.finish();

            prop_assert_eq!(want_hash, got_hash);
        }
    }

    proptest! {
        #[test]
        fn nth_and_rank(
            std_map in btree_map(-64..=128, any::<i32>(), 0..=128),
            key in -128..=128,
        ) {
            let mut map: Treap<i32, i32> = std_map.iter().map(|(k, v)| (*k, *v)).collect();

            let rank = map.rank(&key);
            prop_assert_eq!(rank, map.range(..&key).count());

            let nth = map.nth(rank);
            prop_assert_eq!(nth, map.iter().nth(rank));


            let nth = nth.map(|(k, v)| (*k, *v)); // Clone.
            let removed_nth = map.remove_nth(rank);
            prop_assert_eq!(removed_nth, nth);
            if let Some((key ,_)) = removed_nth {
                prop_assert_eq!(map.get(&key), None);
            }
        }
    }
}
