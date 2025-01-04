// -------------------- CASOS DE TESTE: ES6 (Devem Ser Capturados) --------------------

function ignoreFirst(...[, b, c]) {
  return b + c;
}

// 1. Spread em Chamadas de Função
function sum(a, b, c) {
  return a + b + c;
}
const spreadFirst1 = [1, 2, 3];
console.log(sum(...spreadFirst1)); // Spread em chamada de função

// 2. Rest em Parâmetros de Função
function collect(...restFirst1) {
  return restFirst1;
}
console.log(collect(1, 2, 3)); // Rest em parâmetros de função

// 3. Spread em Arrays
const spreadSecond1 = [1, 2];
const spreadObjFirst1 = [...spreadSecond1, 3, 4]; // Spread em array
console.log(spreadObjFirst1);

// 5. Rest com Parâmetros Nomeados
function processData(restFirst2, restSecond1, ...restThird) {
  console.log(restFirst2, restSecond1, restThird); // Rest após parâmetros nomeados
}
processData(1, 2, 3, 4, 5);

// 6. Desestruturação de Array com Rest (ES6)
const [restFirst3, restSecond2, ...restThird2] = [1, 2, 3, 4, 5]; // Rest em desestruturação de array
console.log(restThird2);

// -------------------- CASOS DE TESTE: ES9 (Não Devem Ser Capturados) --------------------

// 1. Rest em Declarações de Objeto (ES9)
const { restObjFirst4, ...restSecond2 } = { restObjFirst4: 1, restSecond2: 2, restThird: 3 }; // Rest em objetos (ES9)
console.log(restSecond2);

// 2. Spread em Propriedades de Objeto (ES9)
const spreadFirst2 = { restFirst3: 1, restSecond3: 2 };
const spreadObjFirst4 = { ...spreadFirst2 }; // Spread em propriedades de objeto (ES9)
console.log(spreadObjFirst4);

// 3. Spread em Arrays Dentro de Literais (ES9)
const spreadFirst3 = [1, 2];
const spreadSecond3 = [...spreadFirst3, 3, ...[4, 5]]; // Spread aninhado (ES9)
console.log(spreadSecond3);

// 4. Rest em Parâmetros de Arrow Functions (ES9)
const func = (...restObjFirst4) => restObjFirst4; // Rest em arrow function (ES9)
console.log(func(1, 2, 3));

// 5. Spread em Objetos com Propriedades Computadas (ES9)
const dynamic = { [`prop`]: 'value', ...{} }; // Computed property com spread (ES9)
console.log(dynamic);

// 6. Spread/Rest dentro de uma função de callback ou uso posterior de arrays/objetos (ES9)
const spreadObjFirst5 = [1, 2, 3];
const multipliedNumbers = [...spreadObjFirst5].map(x => x * 2); // Spread dentro de função (ES9)
console.log(multipliedNumbers);

const newObject = { ...oldObject, newKey: value }; // Spread dentro de objeto (ES9)

const { ...rest } = oldObject; // Rest

// -------------------- CASOS AMBIGUOS PARA GARANTIR VALIDAÇÃO --------------------

// 1. Strings Contendo a Sintaxe de Spread
const str = "Here is a string with ... syntax"; // Não deve ser capturado
const templateStr = `Template with ... syntax in ES9`; // Não deve ser capturado

function example(...restFirst1Ambiguous) {
  const spreadFirst1 = [1, 2, 3];
  console.log(restFirst1, ...spreadFirst1); // Aqui usamos rest e spread juntos, mas em contextos diferentes
}
example(4, 5, 6); // Passando parâmetros para 'rest' e usando 'spread' para passar elementos de um array

// 2. Comentários com Spread ou Rest
// Example of spread syntax: [...array]
// Example of rest syntax: (...args)

// 3. Propriedades Computadas com Spread
const dynamicSpread = { [`prop`]: 'value', ...{ a: 1 } }; // Computed property com spread (ES9)
console.log(dynamicSpread);

// 4. Uso de Spread/Rest em Funções de Callback
const filterValues = [1, 2, 3, 4].filter(value => value > 2); // Não deve ser capturado
const totalSum = [1, 2, 3, 4].reduce((sum, num) => sum + num, 0); // Não deve ser capturado


// Object property shorthand
const name = 'Raj';
const age = 20;
const location = {age};

// User with ES6 shorthand
// property
const user = {
  name,
  ...location
};

var xhr = { // mock object
  aborted: 0,
  responseText: null,
  responseXML: null,
  status: 0,
  statusText: 'n/a',
  getAllResponseHeaders: function() {},
  getResponseHeader: function() {},
  setRequestHeader: function() {},
  abort: function() {
    log('aborting upload...');
    var e = 'aborted';
    this.aborted = 1;
    $io.attr('src', s.iframeSrc); // abort op in progress
    xhr.error = e;
    s.error && s.error.call(s.context, xhr, 'error', e);
    g && $.event.trigger("ajaxError", [xhr, s, e]);
    s.complete && s.complete.call(s.context, xhr, 'error');
  }
};