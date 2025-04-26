import os

import torch
import torch.distributed as dist

from utils import get_backend


# print_rank_0 - 5 баллов
def print_rank_0(message):
    """
    Зачастую нам хочется что-то вывести, однако весь код выполняется на всех процессах,
    поэтому вывод сообщений дублируется. Допишите функцию print_rank_0, которая
    выводила бы сообщение только на нулевом (главном) процессе.
    
    После этого нужно синхронизировать процессы, т.к. в данном случае мы не хотим,
    чтобы другие процессы уходили в другие функции, а дождались, пока главный процесс
    допечатает сообщение.

    Нужно использовать dist.get_rank и dist.barrier
    """

    if local_rank == 0:
        print(message)

    dist.barrier()
    # raise NotImplemented()



# blocking_send_to_last - 5 баллов
def blocking_send_to_last():
    """
    Ваша послать с каждого процессе, кроме последнего, свой ранг последнему процессу.
    Последний процесс должен получить ранги всех остальных процессов и сложить их.

    Для пересылки нужно использовать блокирующий dist.send.
    Для получения нужно использовать блокирующий dist.recv. Обратите внимание, что в recv
    аргумент src не обязателен!
    
    Документация https://pytorch.org/docs/stable/distributed.html#point-to-point-communication
    
    """
    if local_rank != world_size - 1:
        send_value = torch.Tensor([dist.get_rank()]).long()
        # если не последний процесс, то посылаем свой ранг
        dist.send(send_value, dst=world_size - 1)
    else:
        # если последний процесс, то принимаем от всех остальных
        recv_tensor = torch.zeros(world_size - 1).long()
        for i in range(world_size - 1):
            dist.recv(tensor=recv_tensor[i])
        print(f"Сумма рангов всех процессов: {recv_tensor.sum().item()}")
    dist.barrier()
    print_rank_0("Успешно послали свои ранги последнему процессу")
    



# cyclic send-recv - 5 баллов
def cyclic_send_recv():
    """
    В этой задаче вам необходимо послать значение send_tensor следующему процессу (ранг + 1) от текущего
    процесса и соответственно принять результат посылки от предыдущего процесса (ранг - 1).
    Т.е. 0й посылает тензор 1му, 1й процесс посылает данные 2му, 2й посылает 3му, а 3й посылает 0му.
    Аналогично с получениями: 0й получает от 3го, 1й от 0го, 2й от 1го и 3й от 2го.

    Для посылки и принятия результатов используйте асинхронные функции dist.isend, dist.irecv.
    Эти функции возвращают объект Work, у которого есть метод .wait() - он позволяет дождаться конца
    коммуникации, которая его породила.

    Документация https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend
    
    """
    values_to_send = [10, 20, 30, 40]
    values_to_recv = [40, 10, 20, 30]
    send_tensor = torch.Tensor([values_to_send[local_rank]])
    recv_tensor = torch.zeros_like(send_tensor)
    
    # асинхронная отправка и получение
    send_req = dist.isend(tensor=send_tensor, dst=local_rank + 1 if local_rank != world_size - 1 else 0)
    recv_req = dist.irecv(tensor=recv_tensor)

    # Ждём завершения отправки и получения
    send_req.wait()
    recv_req.wait()

    assert(recv_tensor.long().item() == values_to_recv[local_rank])
    print_rank_0("Процессы успешно получили тензоры соседних процессов!")


# group_comms - 5 баллов
def group_comms(debug: bool=False):
    """
    На каждом ранге гененрируется случайный тензор.
    Ваша задача:
    1. С помощью операции all_reduce найти минимальное значение среди всех local_tensor
    2. Собрать все local_tensor на всех процессах с помощью all_gather и найти минимальное значение
    """
    local_tensor = torch.rand(1)
    
    # Собрать все local_tensor на всех процессах с помощью all_gather и найти минимальное значение
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    min_value = min([el.item() for el in gathered_tensors])
    
    # С помощью операции all_reduce найти минимальное значение среди всех local_tensor
    dist.all_reduce(local_tensor, op=dist.ReduceOp.MIN)

    if debug:
        print_rank_0(f"\tМинимальное значение через all_gather: {min_value} из {torch.tensor(gathered_tensors)}")
        print_rank_0(f"\tМинимальное значение через all_reduce: {local_tensor.item()}")

    print_rank_0("Успешно провели групповые коммуникации!")
    


if __name__ == "__main__":
    # данное задание предполагает запуск в 4 процесса
    # torchrun --nproc-per-node 4 2_communications.py
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)

    print_rank_0("Это сообщение должно быть выведено всего один раз")
    blocking_send_to_last()
    cyclic_send_recv()
    group_comms(debug=True)