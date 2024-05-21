"use strict";
//------------------------------------------------------------------------------
// Property Shorthand
// Shorter syntax for common object property definition idiom.
// http://es6-features.org/#PropertyShorthand
//------------------------------------------------------------------------------

obj = { x, y }

obj = {
  foo (a, b) {
  },
  *quux (x, y) {
  }
}

class Car {
  constructor(make, model) {
    this[make + model] = 'New Car';
  }
}

let uname = 'Anil',  
    udivision = 'First';  
   
let user = {  
   uname,  
   udivision  
};  
console.log(user.uname);  
console.log(user.udivision);

var department = 'dep_name';  
var emp = {  
    id : 102,  
    name : 'Anil',  
    [department]:'Production'  
}  
console.log(emp);

const obj = {
  ['prop_' + Math.random()]: 'value'
};
esse
const obj = {
  user: {
    [prop]: '123 Street'
  }
};

const obj = {
  name: 'Bob',
  [getProperty()]: 30
};


let obj = {
    foo: "bar",
    [ "baz" + quux() ]: 42
}

function createCar(make, model) {
  const car = {};
  car[make + model] = 'New Car';
  return car;
}

const myCar = createCar('Toyota', 'Corolla');
console.log(myCar['ToyotaCorolla']); // Output: 'New Car' 

const person = {
  firstName: 'John',
  lastName: 'Doe',
  getFullName() {
    return `${this.firstName} ${this.lastName}`;
  }
};

const property = 'Name';
console.log(person['getFull' + property]()); // Output: 'John Doe' 


//------------------------------------------------------------------------------
// Method Properties
// Support for method notation in object property definitions, for both regular
//   functions and generator functions.
// http://es6-features.org/#MethodProperties
//------------------------------------------------------------------------------


// Exemplo de definição de propriedades de objetos no ECMAScript 3
var obj = {
    prop1: "foo",
    prop2: "bar",
    method: function() {
      console.log("Método");
    }
  };
  
  console.log(obj.prop1); // Output: foo
  console.log(obj.prop2); // Output: bar
  obj.method(); // Output: Método

// Exemplo de definição de propriedades de objetos no ES5
var prop1 = "foo";
var prop2 = "bar";

var obj = {
  prop1: prop1,
  prop2: prop2,
  method: function() {
    console.log("Método");
  }
};

console.log(obj.prop1); // Output: foo
console.log(obj.prop2); // Output: bar
obj.method(); // Output: Método

var obj = {
  foo: "bar"
};
obj[ "baz" + quux() ] = 42;