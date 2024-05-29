function createPerson(name, age) {
  return {
    name,
    age,
    [`is${name}Adult`]: age >= 18
  };
}

const person = createPerson("John", 25);
console.log(person);
