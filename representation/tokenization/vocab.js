// representation/tokenization/vocab.js

export class Vocab {
  constructor() {
    this.map = new Map();
    this.nextId = 1; // 0 留给 UNK
  }

  getId(key) {
    if (!key) return 0;
    if (!this.map.has(key)) {
      this.map.set(key, this.nextId++);
    }
    return this.map.get(key);
  }

  size() {
    return this.nextId;
  }
}
