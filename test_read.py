import memoria
r = memoria.recordar()
print('documents_count:', len(r.get('documents', [])))
print('documents:', r.get('documents', []))
