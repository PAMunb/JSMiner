// Teste para '*='
let multiplyAssignment = 5;
multiplyAssignment *= 2; // multiplyAssignment = 10
console.log("*= : ", multiplyAssignment);

// Teste para '/='
let divideAssignment = 10;
divideAssignment /= 2; // divideAssignment = 5
console.log("/= : ", divideAssignment);

// Teste para '%='
let modulusAssignment = 10;
modulusAssignment %= 3; // modulusAssignment = 1
console.log("%= : ", modulusAssignment);

// Teste para '+='
let addAssignment = 5;
addAssignment += 10; // addAssignment = 15
console.log("+= : ", addAssignment);

// Teste para '-='
let subtractAssignment = 10;
subtractAssignment -= 5; // subtractAssignment = 5
console.log("-= : ", subtractAssignment);

// Teste para '<<='
let leftShiftAssignment = 5; // 0101
leftShiftAssignment <<= 1; // 1010 (10 em decimal)
console.log("<<= : ", leftShiftAssignment);

// Teste para '>>='
let rightShiftAssignment = 10; // 1010
rightShiftAssignment >>= 1; // 0101 (5 em decimal)
console.log(">>= : ", rightShiftAssignment);

// Teste para '>>>='
let unsignedRightShiftAssignment = -10; // Representação binária: complementos de 2
unsignedRightShiftAssignment >>>= 1; // Desloca e preenche com 0
console.log(">>>= : ", unsignedRightShiftAssignment);

// Teste para '&='
let andAssignment = 5; // 0101
andAssignment &= 3; // 0101 & 0011 = 0001 (1 em decimal)
console.log("&= : ", andAssignment);

// Teste para '^='
let xorAssignment = 5; // 0101
xorAssignment ^= 3; // 0101 ^ 0011 = 0110 (6 em decimal)
console.log("^= : ", xorAssignment);

// Teste para '|='
let orAssignment = 5; // 0101
orAssignment |= 3; // 0101 | 0011 = 0111 (7 em decimal)
console.log("|= : ", orAssignment);

// Teste para '**='
let exponentiationAssignment = 2;
exponentiationAssignment **= 3; // exponentiationAssignment = 2³ = 8
console.log("**= : ", exponentiationAssignment);

// Teste para '??='
let nullishCoalescingAssignment = null;
nullishCoalescingAssignment ??= "Default Value"; // nullishCoalescingAssignment = "Default Value"
console.log("??= : ", nullishCoalescingAssignment);


let a = true;
a &&= false; // a será false, pois `true && false` é false.

let b = null;
b ||= "default"; // a será "default", pois null é falsy.

let c = undefined;
c ??= "default"; // a será "default", pois undefined é nullish.