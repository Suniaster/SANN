use std::rc::Rc;
use std::cell::RefCell;
pub type Container<T> = Rc<RefCell<T>>;

pub fn new_container<T>(obj: T) -> Container<T> {
  return Rc::new(RefCell::new(obj));
}
