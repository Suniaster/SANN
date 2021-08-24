use std::cell::RefCell;
use std::rc::Rc;
pub type Container<T> = Rc<RefCell<T>>;

pub fn new_container<T>(obj: T) -> Container<T> {
    return Rc::new(RefCell::new(obj));
}
