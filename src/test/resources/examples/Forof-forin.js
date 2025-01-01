//////////////////    for of    //////////////////////

const text = "hello";
for (const char of text) {
    console.log(char);
}


const fruits = ["apple", "banana", "cherry"];
for (const fruit of fruits) {
    console.log(fruit);
}

const set = new Set([1, 2, 3]);
for (const number of set) {
    console.log(number);
}

const map = new Map([
    ["key1", "value1"],
    ["key2", "value2"],
]);
for (const [key, value] of map) {
    console.log(`${key}: ${value}`);
}

async function process(array) {
    for await (let i of array) {
      doSomething(i);
    }
  }


//////////////////    for in    //////////////////////

const person = {
    name: "Alice",
    age: 30,
    city: "New York"
};

for (const key in person) {
    console.log(`${key}: ${person[key]}`);
}

const parent = { parentProp: "I am inherited" };
const child = Object.create(parent);
child.ownProp = "I am own property";

for (const key in child) {
    console.log(`${key}: ${child[key]}`);
}

const parent = { inherited: "value" };
const obj = Object.create(parent);
obj.own = "value";

for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
        console.log(`${key}: ${obj[key]}`);
    }
}

const fruits = ["apple", "banana", "cherry"];

for (const index in fruits) {
    console.log(`Index: ${index}, Value: ${fruits[index]}`);
}

const obj = { a: 1, b: 2 };
Object.defineProperty(obj, "hidden", {
    value: 3,
    enumerable: false
});

for (const key in obj) {
    console.log(key);
}

const text = "hello";

for (const index in text) {
    console.log(`Index: ${index}, Character: ${text[index]}`);
}